from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
import psycopg2
from pydantic import BaseModel
import os
import json
import numpy as np
import torch
import requests
import gc # Garbage Collection
from dotenv import load_dotenv

# --- IMPORT MODULES ---
from neural_net import LeadAwareResNet1D
from lead_reconstruction import LeadReconstructionNet
from signal_processing import Signal, compute_hrv_features, detect_r_peaks

load_dotenv()

# --- CONFIGURATION ---
DB_PASSWORD = os.environ.get("DB_PASSWORD")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# Model Config
RESULTS_JSON_PATH = "results.json"
if os.path.exists(RESULTS_JSON_PATH):
    with open(RESULTS_JSON_PATH, "r") as f:
        meta = json.load(f)
        CLASS_NAMES = meta["per_class_metrics"]["class_names"]
        arch = meta["model_architecture"]["architecture"]
        CLF_PARAMS = {
            "n_leads": arch["num_leads"],
            "num_labels": arch["num_classes"],
            "base": arch["base_channels"],
            "depths": tuple(arch["depths"]),
            "k": arch["kernel_size"],
            "use_lead_mixer": True,
            "use_rhythm_head": True
        }
else:
    # Fallback
    CLASS_NAMES = ["Error"]
    CLF_PARAMS = {}

CLF_MODEL_URL = os.environ.get("CLF_MODEL_URL")
RECON_MODEL_URL = os.environ.get("RECON_MODEL_URL")
CLF_PATH = "model_weights.pth"
RECON_PATH = "lead_reconstruction.pth"

# Constants
TARGET_FS = 500.0
BP_LOW = 0.5
BP_HIGH = 40.0

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
device = torch.device("cpu")

# --- MEMORY MANAGEMENT ---
def clear_memory():
    """Forces garbage collection to free RAM."""
    gc.collect()

def load_classifier():
    """Loads classifier only when needed."""
    if not os.path.exists(CLF_PATH):
        download_file(CLF_MODEL_URL, CLF_PATH)
    
    print("🧠 Loading Classifier...")
    model = LeadAwareResNet1D(**CLF_PARAMS).to(device)
    ckpt = torch.load(CLF_PATH, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model

def load_reconstructor():
    """Loads reconstructor only when needed."""
    if not os.path.exists(RECON_PATH):
        download_file(RECON_MODEL_URL, RECON_PATH)
        
    print("🎨 Loading Reconstructor...")
    recon = LeadReconstructionNet().to(device)
    recon.load_state_dict(torch.load(RECON_PATH, map_location=device))
    recon.eval()
    return recon

def download_file(url, dest):
    if not url or os.path.exists(dest): return
    print(f"📥 Downloading {dest}...")
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Error downloading: {e}")

# --- STARTUP (ONLY DOWNLOAD, DON'T LOAD TO RAM) ---
@app.on_event("startup")
async def startup_event():
    # Only download files to disk. Do NOT load models into RAM yet.
    download_file(CLF_MODEL_URL, CLF_PATH)
    download_file(RECON_MODEL_URL, RECON_PATH)

# --- SECURITY & DATA ---
async def verify_api_key(key: str = Security(api_key_header)):
    if not API_SECRET_KEY: return True
    if key == API_SECRET_KEY: return True
    raise HTTPException(status_code=403, detail="Invalid API Key")

def fetch_and_process_ecg(user_id, start, end, cursor):
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
    if filtered.shape[1] < 5000:
        pad = np.zeros((3, 5000 - filtered.shape[1]), dtype=np.float32)
        filtered = np.concatenate([filtered, pad], axis=1)
    return filtered[:, :5000]

# --- ENDPOINTS ---
class AnalysisRequest(BaseModel):
    user_id: str
    start: str
    end: str

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
def analyze_data(request: AnalysisRequest):
    try:
        clear_memory() # Free up RAM before starting
        
        # 1. Fetch Data
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                signal = fetch_and_process_ecg(request.user_id, request.start, request.end, cursor)
                if signal is None: return {"accepted": False, "error": "No data"}

                # 2. Features
                lead_ii = signal[1, :]
                peaks = detect_r_peaks(lead_ii, fs=TARGET_FS)
                hrv = compute_hrv_features(peaks, fs=TARGET_FS)

                # 3. Load Model -> Inference -> Unload
                clf_model = load_classifier() # Load just-in-time
                
                x = torch.from_numpy(signal).unsqueeze(0).to(device)
                h = torch.from_numpy(hrv).unsqueeze(0).float().to(device)
                a = torch.tensor([0.5], device=device).float()
                s = torch.tensor([0.5], device=device).float()

                with torch.no_grad():
                    logits = clf_model(x, hrv_features=h, age=a, sex=s)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # UNLOAD MODEL IMMEDIATELY
                del clf_model
                clear_memory() 

                top_idx = int(np.argmax(probs))
                pred = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
                conf = float(probs[top_idx])
                all_probs = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}

                # 4. Audit Log
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
        clear_memory()
        return {"accepted": False, "error": str(e)}

@app.post("/reconstruct", dependencies=[Depends(verify_api_key)])
def reconstruct_data(request: AnalysisRequest):
    try:
        clear_memory()
        
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                signal = fetch_and_process_ecg(request.user_id, request.start, request.end, cursor)
                if signal is None: return {"accepted": False, "error": "No data"}

                # Load Model -> Inference -> Unload
                recon_model = load_reconstructor()
                
                x = torch.from_numpy(signal).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = recon_model(x)
                recon_data = out.cpu().numpy()[0].tolist()
                
                # UNLOAD
                del recon_model
                clear_memory()

                cursor.execute(
                    "INSERT INTO ecg_analysis_results (user_id, start_ts, end_ts, metrics) VALUES (%s, %s, %s, %s) RETURNING id",
                    (request.user_id, request.start, request.end, json.dumps({"type": "reconstruction", "status": "success"}))
                )
                job_id = cursor.fetchone()[0]
                conn.commit()

        return {"accepted": True, "job_id": str(job_id), "reconstruction": recon_data}

    except Exception as e:
        clear_memory()
        return {"accepted": False, "error": str(e)}