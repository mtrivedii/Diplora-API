import modal
import torch
import numpy as np
import os
import json
import time
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client

# Import your actual model definitions
from neural_net import LeadAwareResNet1D
from lead_reconstruction import LeadReconstructionNet

# 1. Define the Modal Environment
# We reuse the requirements from your training setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("supabase")
)

app = modal.App("diplora-api-production")
web_app = FastAPI(title="Diplora AI Analysis Service")

# Mount the volume where your trained weights live
v_results = modal.Volume.from_name("results")

# 2. Input Data Model (The "Trigger" Payload)
class AnalysisRequest(BaseModel):
    job_id: str       # UUID for traceability
    record_id: int    # ID of the measurement in Supabase
    user_id: str      # For permission checks/logging

@app.cls(
    image=image,
    gpu="T4",  # T4 is perfect for inference (fast & cheap)
    volumes={"/v_results": v_results},
    secrets=[modal.Secret.from_name("diplora-secrets")], # Needs SUPABASE_URL, SUPABASE_KEY, API_SECRET
    keep_warm=1, # crucial: keeps 1 container ready to respond instantly
    concurrency_limit=10 # scale up as needed
)
class ModelService:
    def __enter__(self):
        """
        Cold Start Initialization: Runs once when container spins up.
        Loads config and weights so we don't do it per-request.
        """
        print("⚡ Initializing AI Service...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- A. Load Configuration dynamically ---
        # We look for the latest run or a specific 'production' folder
        # For now, let's assume you copy your best run to a fixed path 'model_export_3lead' in the volume
        # Or we can scan for it. Let's use a fixed path for reliability.
        model_dir = "/v_results/model_export_3lead" 
        
        if not os.path.exists(model_dir):
            # Fallback: try to find the latest timestamped folder
            all_dirs = sorted([d for d in os.listdir("/v_results") if os.path.isdir(os.path.join("/v_results", d))])
            if all_dirs:
                model_dir = os.path.join("/v_results", all_dirs[-1])
                print(f"⚠️ Using latest run found: {model_dir}")
        
        print(f"📂 Loading models from: {model_dir}")
        
        # Load Metadata
        with open(os.path.join(model_dir, "results.json"), "r") as f:
            self.meta = json.load(f)
            
        arch = self.meta["model_architecture"]["architecture"]
        self.class_names = self.meta["per_class_metrics"]["class_names"]
        
        # --- B. Load Classifier ---
        self.classifier = LeadAwareResNet1D(
            n_leads=arch["num_leads"],
            num_labels=arch["num_classes"],
            base=arch["base_channels"],
            depths=tuple(arch["depths"]),
            k=arch["kernel_size"],
            p_drop=arch["dropout"],
            use_lead_mixer=True,   # Explicitly setting based on your config
            use_rhythm_head=True   # Explicitly setting based on your config
        ).to(self.device)
        
        clf_weights = os.path.join(model_dir, "model_weights.pth")
        self.classifier.load_state_dict(torch.load(clf_weights, map_location=self.device))
        self.classifier.eval()
        
        # --- C. Load Reconstructor ---
        # (Assuming standard architecture for recon as per lead_reconstruction.py)
        self.recon = LeadReconstructionNet().to(self.device)
        recon_weights = os.path.join(model_dir, "lead_reconstruction.pth")
        if os.path.exists(recon_weights):
            self.recon.load_state_dict(torch.load(recon_weights, map_location=self.device))
            self.recon.eval()
        else:
            print("⚠️ Reconstruction weights not found, skipping recon.")
            self.recon = None

        # --- D. Connect to Supabase ---
        self.supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        print("✅ Service Ready & Connected")

    def fetch_signal(self, record_id: int):
        """Fetch raw ECG data from Supabase"""
        resp = self.supabase.table("measurements").select("*").eq("id", record_id).execute()
        if not resp.data:
            raise ValueError(f"Record {record_id} not found")
        
        row = resp.data[0]
        
        # Convert to numpy [3, 5000] (Lead I, II, V2)
        # Note: Ensure your DB columns match these keys exactly
        try:
            sig = np.stack([
                np.array(row["lead_i"]),
                np.array(row["lead_ii"]), 
                np.array(row["lead_v2"])
            ])
        except KeyError:
            # Fallback if names differ (e.g. channel1, channel2)
            sig = np.stack([
                np.array(row.get("channel1", [])),
                np.array(row.get("channel2", [])), 
                np.array(row.get("channel3", []))
            ])
            
        # Basic Validation
        if sig.shape[1] < 100:
            raise ValueError("Signal too short")
            
        # Normalize (Important: Matches training preprocessing)
        # (x - mean) / std
        m = sig.mean(axis=1, keepdims=True)
        s = sig.std(axis=1, keepdims=True) + 1e-8
        sig = (sig - m) / s
        
        return torch.from_numpy(sig).float().unsqueeze(0) # Add batch dim [1, 3, T]

    @modal.web_endpoint(method="POST")
    def analyze(self, request: AnalysisRequest, x_api_key: str = Header(None)):
        """
        The Secure Endpoint called by Supabase Edge Functions
        """
        # 1. ISO 13485 Security Gate
        if x_api_key != os.environ["API_SECRET"]:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        start_time = time.time()
        
        try:
            # 2. Data Acquisition
            x = self.fetch_signal(request.record_id).to(self.device)
            
            # 3. Inference (Classification)
            # Create dummy features for rhythm head if needed (batch=1)
            # Ideally, you'd calculate these real-time using signal_processing.py
            hrv_dummy = torch.zeros((1, 12)).to(self.device) 
            age_dummy = torch.tensor([0.5]).to(self.device) # Default normalized age
            sex_dummy = torch.tensor([0.5]).to(self.device) # Default unknown sex
            
            with torch.no_grad():
                logits = self.classifier(x, hrv_features=hrv_dummy, age=age_dummy, sex=sex_dummy)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Get top prediction
                top_idx = int(np.argmax(probs))
                prediction = self.class_names[top_idx]
                confidence = float(probs[top_idx])
                
                # Reconstruction (Optional)
                recon_data = None
                if self.recon:
                    recon_out = self.recon(x)
                    recon_data = recon_out.cpu().numpy()[0].tolist()

            # 4. Traceability Logging (Write back to DB)
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
            
            self.supabase.table("analysis_results").insert(payload).execute()
            
            return {
                "status": "success", 
                "prediction": prediction,
                "confidence": confidence
            }

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            # Log failure for audit trail
            self.supabase.table("analysis_results").insert({
                "job_id": request.job_id,
                "measurement_id": request.record_id,
                "status": "failed",
                "error_log": str(e)
            }).execute()
            raise HTTPException(status_code=500, detail=str(e))