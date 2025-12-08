import modal
import torch
import numpy as np
import os
import json
import time
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client
from huggingface_hub import hf_hub_download

from neural_net import LeadAwareResNet1D
from lead_reconstruction import LeadReconstructionNet

# 1. Define Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("supabase", "huggingface_hub")
    .add_local_file("neural_net.py", remote_path="/root/neural_net.py")
    .add_local_file("lead_reconstruction.py", remote_path="/root/lead_reconstruction.py")
    .add_local_file("signal_processing.py", remote_path="/root/signal_processing.py")
    .add_local_file("filter.py", remote_path="/root/filter.py")
)

app = modal.App("diplora-api-production")
web_app = FastAPI(title="Diplora AI Analysis Service")

class AnalysisRequest(BaseModel):
    job_id: str
    record_id: int
    user_id: str

@app.cls(
    image=image,
    gpu="T4",
    # IMPORTANT: Ensure HF_TOKEN is in this secret!
    secrets=[modal.Secret.from_name("diplora-secrets")], 
    keep_warm=1
)
class ModelService:
    def __enter__(self):
        print("⚡ Initializing AI Service...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        HF_REPO_ID = "maanit/diplora-demo"
        # Modal will automatically pull this from the secret "diplora-secrets"
        HF_TOKEN = os.environ.get("HF_TOKEN") 

        HF_FILES = {
            "config": "results.json",
            "classifier": "model_weights.pth", # <--- Corrected
            "reconstructor": "lead_reconstruction.pth"
        }
        
        # --- A. Load Configuration ---
        print(f"📥 Fetching config from Hugging Face: {HF_REPO_ID}...")
        try:
            config_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILES["config"], token=HF_TOKEN)
            with open(config_path, "r") as f:
                self.meta = json.load(f)
            
            arch = self.meta["model_architecture"]["architecture"]
            self.class_names = self.meta["per_class_metrics"]["class_names"]
            print(f"✅ Config loaded: {len(self.class_names)} classes")
        except Exception as e:
            print(f"❌ Failed to load results.json: {e}")
            # FALLBACK defaults for Modal
            self.class_names = ["AF", "AFL", "NSR", "Other"] # (Truncated for brevity, full list in main.py)
            arch = {"num_leads": 3, "num_classes": 23, "base_channels": 48, "depths": [3,6,12,4], "kernel_size": 7, "dropout": 0.3}

        # --- B. Load Classifier ---
        print("🧠 Loading Classifier...")
        clf_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILES["classifier"], token=HF_TOKEN)
        
        self.classifier = LeadAwareResNet1D(
            n_leads=arch["num_leads"],
            num_labels=arch["num_classes"],
            base=arch["base_channels"],
            depths=tuple(arch["depths"]),
            k=arch["kernel_size"],
            p_drop=arch["dropout"],
            use_lead_mixer=True,
            use_rhythm_head=True
        ).to(self.device)
        
        self.classifier.load_state_dict(torch.load(clf_path, map_location=self.device))
        self.classifier.eval()
        
        # --- C. Load Reconstructor ---
        print("🎨 Loading Reconstructor...")
        try:
            recon_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILES["reconstructor"], token=HF_TOKEN)
            self.recon = LeadReconstructionNet().to(self.device)
            self.recon.load_state_dict(torch.load(recon_path, map_location=self.device))
            self.recon.eval()
        except Exception as e:
            print(f"⚠️ Reconstruction weights not found: {e}")
            self.recon = None

        self.supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    def fetch_signal(self, record_id: int):
        resp = self.supabase.table("measurements").select("*").eq("id", record_id).execute()
        if not resp.data:
            raise ValueError(f"Record {record_id} not found in Supabase")
        
        row = resp.data[0]
        try:
            sig = np.stack([
                np.array(row.get("lead_i") or row.get("channel1")),
                np.array(row.get("lead_ii") or row.get("channel2")),
                np.array(row.get("lead_v2") or row.get("channel3"))
            ])
        except Exception:
            raise ValueError("Could not find lead data columns")

        m = sig.mean(axis=1, keepdims=True)
        s = sig.std(axis=1, keepdims=True) + 1e-8
        sig = (sig - m) / s
        
        return torch.from_numpy(sig).float().unsqueeze(0) 

    @modal.web_endpoint(method="POST")
    def analyze(self, request: AnalysisRequest, x_api_key: str = Header(None)):
        if x_api_key != os.environ["API_SECRET"]:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        start_time = time.time()
        try:
            x = self.fetch_signal(request.record_id).to(self.device)
            
            hrv_dummy = torch.zeros((1, 12)).to(self.device)
            age_dummy = torch.tensor([0.5]).to(self.device)
            sex_dummy = torch.tensor([0.5]).to(self.device)

            with torch.no_grad():
                logits = self.classifier(x, hrv_features=hrv_dummy, age=age_dummy, sex=sex_dummy)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                recon_data = None
                if self.recon:
                    recon_out = self.recon(x)
                    recon_data = recon_out.cpu().numpy()[0].tolist()

            top_idx = int(np.argmax(probs))
            prediction = self.class_names[top_idx] if top_idx < len(self.class_names) else str(top_idx)
            confidence = float(probs[top_idx])
            
            payload = {
                "job_id": request.job_id,
                "measurement_id": request.record_id,
                "status": "completed",
                "prediction": prediction,
                "confidence": confidence,
                "full_probabilities": {name: float(p) for name, p in zip(self.class_names, probs)},
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "model_version": "v1.0-hf-private"
            }
            
            self.supabase.table("analysis_results").insert(payload).execute()
            
            return {
                "status": "success", 
                "prediction": prediction, 
                "confidence": confidence
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            self.supabase.table("analysis_results").insert({
                "job_id": request.job_id, 
                "measurement_id": request.record_id, 
                "status": "failed", 
                "error_log": str(e)
            }).execute()
            raise HTTPException(status_code=500, detail=str(e))