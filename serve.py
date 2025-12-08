import modal
import torch
import numpy as np
import os
import json
import time
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client

# Import your model definitions (ensure these files are in the same folder)
from neural_net import LeadAwareResNet1D
from lead_reconstruction import LeadReconstructionNet

# 1. Define the Modal Environment
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("supabase")
    # Add your local model files to the container
    .add_local_file("neural_net.py", remote_path="/root/neural_net.py")
    .add_local_file("lead_reconstruction.py", remote_path="/root/lead_reconstruction.py")
    .add_local_file("signal_processing.py", remote_path="/root/signal_processing.py")
    .add_local_file("filter.py", remote_path="/root/filter.py")
)

app = modal.App("diplora-api-production")
web_app = FastAPI(title="Diplora AI Analysis Service")

# Mount the volume where your trained weights live
v_results = modal.Volume.from_name("results")

# 2. Input Data Model
class AnalysisRequest(BaseModel):
    job_id: str       # UUID for traceability
    record_id: int    # ID of the measurement in Supabase
    user_id: str      # For permission checks

@app.cls(
    image=image,
    gpu="T4",  # T4 is cheap and fast enough for inference
    volumes={"/v_results": v_results},
    secrets=[modal.Secret.from_name("diplora-secrets")], # Needs SUPABASE_URL, SUPABASE_KEY, API_SECRET
    keep_warm=1
)
class ModelService:
    def __enter__(self):
        """Run once on container startup to load models (Cold Start Optimization)"""
        print("⚡ Initializing AI Service...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- A. Load Configuration ---
        # We look for the folder containing your specific run
        # Update this if your folder name is different in the Volume!
        model_dir = "/v_results/model_export_3lead" 
        
        # Fallback: Find latest run if specific folder doesn't exist
        if not os.path.exists(model_dir):
            print(f"⚠️ '{model_dir}' not found. Scanning for latest run...")
            all_dirs = sorted([d for d in os.listdir("/v_results") if os.path.isdir(os.path.join("/v_results", d))])
            if all_dirs:
                model_dir = os.path.join("/v_results", all_dirs[-1])
        
        print(f"📂 Loading from: {model_dir}")
        
        # Load Metadata to get exact architecture
        try:
            with open(os.path.join(model_dir, "results.json"), "r") as f:
                self.meta = json.load(f)
            
            arch = self.meta["model_architecture"]["architecture"]
            self.class_names = self.meta["per_class_metrics"]["class_names"]
            print(f"✅ Config loaded: {len(self.class_names)} classes, Depths: {arch['depths']}")
        except Exception as e:
            print(f"❌ Failed to load results.json: {e}")
            raise e

        # --- B. Load Classifier ---
        self.classifier = LeadAwareResNet1D(
            n_leads=arch["num_leads"],
            num_labels=arch["num_classes"], # Should be 23
            base=arch["base_channels"],     # Should be 48
            depths=tuple(arch["depths"]),   # Should be [3, 6, 12, 4]
            k=arch["kernel_size"],
            p_drop=arch["dropout"],
            use_lead_mixer=True,  # Assuming True based on training
            use_rhythm_head=True  # Assuming True based on training
        ).to(self.device)
        
        clf_path = os.path.join(model_dir, "model_weights.pth")
        self.classifier.load_state_dict(torch.load(clf_path, map_location=self.device))
        self.classifier.eval()
        
        # --- C. Load Reconstructor ---
        self.recon = LeadReconstructionNet().to(self.device)
        recon_path = os.path.join(model_dir, "lead_reconstruction.pth")
        
        if os.path.exists(recon_path):
            self.recon.load_state_dict(torch.load(recon_path, map_location=self.device))
            self.recon.eval()
            print("✅ Reconstruction model loaded")
        else:
            print("⚠️ Reconstruction weights not found, skipping.")
            self.recon = None

        # --- D. Connect to Supabase ---
        self.supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    def fetch_signal(self, record_id: int):
        """Fetch raw ECG from Supabase"""
        resp = self.supabase.table("measurements").select("*").eq("id", record_id).execute()
        if not resp.data:
            raise ValueError(f"Record {record_id} not found in Supabase")
        
        row = resp.data[0]
        
        # Stack leads (ensure DB column names match these!)
        # Adjust 'lead_i', 'lead_ii' if your DB uses different names
        try:
            sig = np.stack([
                np.array(row.get("lead_i") or row.get("channel1")),
                np.array(row.get("lead_ii") or row.get("channel2")),
                np.array(row.get("lead_v2") or row.get("channel3"))
            ])
        except Exception:
            raise ValueError("Could not find lead data columns (lead_i/channel1) in record")

        # Normalize (Standardize)
        m = sig.mean(axis=1, keepdims=True)
        s = sig.std(axis=1, keepdims=True) + 1e-8
        sig = (sig - m) / s
        
        return torch.from_numpy(sig).float().unsqueeze(0) # [1, 3, T]

    @modal.web_endpoint(method="POST")
    def analyze(self, request: AnalysisRequest, x_api_key: str = Header(None)):
        """
        Secure Endpoint: Triggered by Supabase Edge Function
        """
        if x_api_key != os.environ["API_SECRET"]:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        start_time = time.time()
        
        try:
            # 1. Fetch & Preprocess
            x = self.fetch_signal(request.record_id).to(self.device)
            
            # 2. Inference
            # Create dummy features for rhythm head (required by model forward pass)
            hrv_dummy = torch.zeros((1, 12)).to(self.device)
            age_dummy = torch.tensor([0.5]).to(self.device)
            sex_dummy = torch.tensor([0.5]).to(self.device)

            with torch.no_grad():
                logits = self.classifier(x, hrv_features=hrv_dummy, age=age_dummy, sex=sex_dummy)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Reconstruction
                recon_data = None
                if self.recon:
                    recon_out = self.recon(x)
                    recon_data = recon_out.cpu().numpy()[0].tolist()

            # 3. Format Results
            top_idx = int(np.argmax(probs))
            prediction = self.class_names[top_idx]
            confidence = float(probs[top_idx])
            
            # 4. Traceability: Log to Supabase.pdf]
            payload = {
                "job_id": request.job_id,
                "measurement_id": request.record_id,
                "status": "completed",
                "prediction": prediction,
                "confidence": confidence,
                "full_probabilities": {name: float(p) for name, p in zip(self.class_names, probs)},
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "model_version": "v1.0-resnet-23class"
            }
            
            # Insert into 'analysis_results' table
            self.supabase.table("analysis_results").insert(payload).execute()
            
            return {
                "status": "success", 
                "prediction": prediction, 
                "confidence": confidence
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            # Log failure
            self.supabase.table("analysis_results").insert({
                "job_id": request.job_id, 
                "measurement_id": request.record_id, 
                "status": "failed", 
                "error_log": str(e)
            }).execute()
            raise HTTPException(status_code=500, detail=str(e))