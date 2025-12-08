from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
import psycopg2
from pydantic import BaseModel
import os
import json
import numpy as np
import torch
import requests
from dotenv import load_dotenv

# --- IMPORT YOUR MODULES ---
from neural_net import LeadAwareResNet1D
from lead_reconstruction import LeadReconstructionNet
from signal_processing import Signal, compute_hrv_features, detect_r_peaks

# --- CONFIGURATION ---
load_dotenv()

# 1. SECRETS
DB_PASSWORD = os.environ.get("DB_PASSWORD")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# 2. MODEL CONFIGURATION (Dynamic Loading)
RESULTS_JSON_PATH = "results.json"  # Ensure this is in your repo!

if not os.path.exists(RESULTS_JSON_PATH):
    print("❌ CRITICAL: results.json not found. API cannot start.")
    # Fallback defaults just to prevent immediate crash during build
    CLASS_NAMES = ["Error: Missing JSON"]
    CLF_PARAMS = {"n_leads": 3, "num_labels": 1, "base": 48, "depths": (3, 6, 12, 4), "k": 7}
else:
    with open(RESULTS_JSON_PATH, "r") as f:
        meta = json.load(f)
        # Load Class Names dynamically
        CLASS_NAMES = meta["per_class_metrics"]["class_names"]
        # Load Model Hyperparameters dynamically
        arch = meta["model_architecture"]["architecture"]
        CLF_PARAMS = {
            "n_leads": arch["num_leads"],
            "num_labels": arch["num_classes"],
            "base": arch["base_channels"],
            "depths": tuple(arch["depths"]),
            "k": arch["kernel_size"],
            "use_lead_mixer": True,   # Defaulting to True based on your files
            "use_rhythm_head": True
        }
        print(f"✅ Config loaded: {len(CLASS_NAMES)} classes.")

# 3. WEIGHT FILES
CLF_MODEL_URL = os.environ.get("CLF_MODEL_URL")     # URL for model_weights.pth
RECON_MODEL_URL = os.environ.get("RECON_MODEL_URL") # URL for lead_reconstruction.pth
CLF_PATH = "model_weights.pth"
RECON_PATH = "lead_reconstruction.pth"

# Preprocessing Constants
TARGET_FS = 500.0
BP_LOW = 0.5
BP_HIGH = 40.0

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global Models
clf_model = None
recon_model = None
device = torch.device("cpu") 

# --- HELPERS ---
def download_file(url, dest):
    if not url: return
    if not os.path.exists(dest):
        print(f"📥 Downloading {dest}...")
        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Download complete.")
            else:
                print(f"⚠️ Failed to download {dest}: {r.status_code}")
        except Exception as e:
            print(f"⚠️ Error downloading {dest}: {e}")

@app.on_event("startup")
async def startup_event():
    global clf_model, recon_model
    
    # Download weights
    download_file(CLF_MODEL_URL, CLF_PATH)
    download_file(RECON_MODEL_URL, RECON_PATH)
    
    # Load Classifier
    try:
        model = LeadAwareResNet1D(**CLF_PARAMS).to(device)
        if os.path.exists(CLF_PATH):
            ckpt = torch.load(CLF_PATH, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state, strict=False)
            model.eval()
            clf_model = model
            print("✅ Classifier Ready")
        else:
            print("⚠️ Classifier weights missing (Check CLF_MODEL_URL)")
    except Exception as e:
        print(f"❌ Classifier Error: {e}")

    # Load Reconstructor
    try:
        recon = LeadReconstructionNet().to(device) 
        if os.path.exists(RECON_PATH):
            recon.load_state_dict(torch.load(RECON_PATH, map_location=device))
            recon.eval()
            recon_model = recon
            print("✅ Reconstructor Ready")
        else:
            print("⚠️ Reconstructor weights missing (Check RECON_MODEL_URL)")
    except Exception as e:
        print(f"❌ Reconstructor Error: {e}")

async def verify_api_key(key: str = Security(api_key_header)):
    if not API_SECRET_KEY: return True 
    if key == API_SECRET_KEY: return True
    raise HTTPException(status_code=403, detail="Invalid API Key")

def fetch_and_process_ecg(user_id, start, end, cursor):
    """Fetches 3 leads (I, II, V2) and filters them."""
    # Updated query to match your schema
    query = """
        SELECT COALESCE(channel1,0), COALESCE(channel2,0), COALESCE(channel3,0)
        FROM ecg_data
        WHERE user_id = %s AND timestamp >= %s AND timestamp < %s
        ORDER BY timestamp ASC LIMIT 5000;
    """
    cursor.execute(query, (user_id, start, end))
    rows = cursor.fetchall()
    if not rows: return None
    
    raw = np.array(rows, dtype=np.float32).T 
    filtered = Signal._bp_filter_np(raw, fs=TARGET_FS, lowcut=BP_LOW, highcut=BP_HIGH)
    
    # Pad to 5000 if short
    if filtered.shape[1] < 5000:
        pad = np.zeros((3, 5000 - filtered.shape[1]), dtype=np.float32)
        filtered = np.concatenate([filtered, pad], axis=1)
    
    return filtered[:, :5000]

# --- API ROUTES ---

class AnalysisRequest(BaseModel):
    user_id: str
    start: str
    end: str

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
def analyze_data(request: AnalysisRequest):
    """Route 1: Classification"""
    global clf_model
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                signal = fetch_and_process_ecg(request.user_id, request.start, request.end, cursor)
                if signal is None: return {"accepted": False, "error": "No data found"}

                # Feature Extraction
                lead_ii = signal[1, :]
                peaks = detect_r_peaks(lead_ii, fs=TARGET_FS)
                hrv = compute_hrv_features(peaks, fs=TARGET_FS)

                # Inference
                if clf_model:
                    x = torch.from_numpy(signal).unsqueeze(0).to(device)
                    h = torch.from_numpy(hrv).unsqueeze(0).float().to(device)
                    a = torch.tensor([0.5], device=device).float() # Dummy age
                    s = torch.tensor([0.5], device=device).float() # Dummy sex

                    with torch.no_grad():
                        logits = clf_model(x, hrv_features=h, age=a, sex=s)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    
                    top_idx = int(np.argmax(probs))
                    # Use the dynamically loaded CLASS_NAMES
                    pred = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"Class {top_idx}"
                    conf = float(probs[top_idx])
                    all_probs = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
                else:
                    return {"accepted": False, "error": "Classifier model not loaded"}

                # Traceable Logging
                payload = {
                    "type": "classification",
                    "prediction": pred,
                    "confidence": conf,
                    "probabilities": all_probs
                }
                cursor.execute(
                    "INSERT INTO ecg_analysis_results (user_id, start_ts, end_ts, metrics, annotations) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (request.user_id, request.start, request.end, json.dumps(payload), json.dumps(list(all_probs.keys())))
                )
                job_id = cursor.fetchone()[0]
                conn.commit()
                
        return {"accepted": True, "job_id": str(job_id), "prediction": pred, "confidence": conf}

    except Exception as e:
        return {"accepted": False, "error": str(e)}

@app.post("/reconstruct", dependencies=[Depends(verify_api_key)])
def reconstruct_data(request: AnalysisRequest):
    """Route 2: 12-Lead Reconstruction"""
    global recon_model
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                signal = fetch_and_process_ecg(request.user_id, request.start, request.end, cursor)
                if signal is None: return {"accepted": False, "error": "No data found"}

                if recon_model:
                    x = torch.from_numpy(signal).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = recon_model(x)
                    recon_data = out.cpu().numpy()[0].tolist()
                else:
                    return {"accepted": False, "error": "Reconstruction model not loaded"}

                # Log the job (Audit Trail)
                cursor.execute(
                    "INSERT INTO ecg_analysis_results (user_id, start_ts, end_ts, metrics) VALUES (%s, %s, %s, %s) RETURNING id",
                    (request.user_id, request.start, request.end, json.dumps({"type": "reconstruction", "status": "success"}))
                )
                job_id = cursor.fetchone()[0]
                conn.commit()

        return {"accepted": True, "job_id": str(job_id), "reconstruction": recon_data}

    except Exception as e:
        return {"accepted": False, "error": str(e)}