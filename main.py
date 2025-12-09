from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
import psycopg2
from pydantic import BaseModel
import os
import json
import numpy as np
import torch
import gc 
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from datetime import datetime # <-- NEW: Used for parsing timestamps

# --- IMPORT MODULES ---
from neural_net import LeadAwareResNet1D
from lead_reconstruction import LeadReconstructionNet
from signal_processing import Signal, compute_hrv_features, detect_r_peaks

load_dotenv()

# --- CONFIGURATION ---
DB_PASSWORD = os.environ.get("DB_PASSWORD")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD or ''}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# --- HUGGING FACE CONFIG ---
HF_REPO_ID = "maanit/diplora-demo"
HF_TOKEN = os.environ.get("HF_TOKEN")

HF_FILES = {
    "config": "results.json",
    "classifier": "model_weights.pth",
    "reconstructor": "lead_reconstruction.pth"
}

# --- FALLBACK CONFIG ---
DEFAULT_ARCH = {
    "n_leads": 3, "num_labels": 23, "base": 48, 
    "depths": (3, 6, 12, 4), "k": 7, 
    "use_lead_mixer": True, "use_rhythm_head": True
}
DEFAULT_CLASSES = [
    "AF", "AFL", "Brady/Escape", "Chamber/axis abnormality", "Fascicular block", 
    "IVCD-other", "LBBB", "LVH", "NSR", "PR", "PSVT", "PVT", "RBBB", "RVH", 
    "SA", "SB", "STach", "SVPB/PAC", "SVT", "VBig", "VEB", "VF", "VTach"
]

# Constants
TARGET_FS = 500.0
BP_LOW = 0.5
BP_HIGH = 40.0

# --- APP SETUP ---
app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
device = torch.device("cpu")

# --- CORE HELPERS (Model Loading & Config) ---
def clear_memory():
    """Forces garbage collection to free RAM."""
    gc.collect()

def load_config():
    """Fetches results.json or falls back to defaults"""
    try:
        print("📥 Fetching config from Hugging Face...")
        config_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILES["config"], token=HF_TOKEN)
        
        with open(config_path, "r") as f:
            meta = json.load(f)
            
        return {
            "class_names": meta["per_class_metrics"]["class_names"],
            "params": {
                "n_leads": meta["model_architecture"]["architecture"]["num_leads"],
                "num_labels": meta["model_architecture"]["architecture"]["num_classes"],
                "base": meta["model_architecture"]["architecture"]["base_channels"],
                "depths": tuple(meta["model_architecture"]["architecture"]["depths"]),
                "k": meta["model_architecture"]["architecture"]["kernel_size"],
                "use_lead_mixer": True,
                "use_rhythm_head": True
            }
        }
    except Exception as e:
        print(f"⚠️ Config download failed ({e}). Using FALLBACK defaults.")
        return {
            "class_names": DEFAULT_CLASSES,
            "params": DEFAULT_ARCH
        }

# Load config immediately (Executes only once on import)
CONFIG_DATA = load_config()
CLASS_NAMES = CONFIG_DATA["class_names"]
CLF_PARAMS = CONFIG_DATA["params"]

def load_classifier():
    """Loads classifier from Private HF Repo (LAZY LOAD)"""
    print("🧠 Loading Classifier...")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILES["classifier"], token=HF_TOKEN)
    
    model = LeadAwareResNet1D(**CLF_PARAMS).to(device)
    ckpt = torch.load(model_path, map_location=device)
    
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model

def load_reconstructor():
    """Loads reconstructor from Private HF Repo (LAZY LOAD)"""
    print("🎨 Loading Reconstructor...")
    recon_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILES["reconstructor"], token=HF_TOKEN)
    
    recon = LeadReconstructionNet().to(device)
    recon.load_state_dict(torch.load(recon_path, map_location=device))
    recon.eval()
    return recon

# --- DYNAMIC LOGIC & DATA ACCESS (MOVED UP) ---

async def verify_api_key(key: str = Security(api_key_header)):
    """API Key verification function for endpoint dependencies."""
    if not API_SECRET_KEY: return True
    if key == API_SECRET_KEY: return True
    raise HTTPException(status_code=403, detail="Invalid API Key")

def get_partition_name(timestamp_str):
    """Parses timestamp string to construct the partition table name."""
    try:
        # Assumes format 'YYYY-MM-DD HH:MM:SS' 
        dt = datetime.strptime(timestamp_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
        return f"ecg_data_y{dt.strftime('%Y')}m{dt.strftime('%m')}"
    except ValueError:
        # Fallback if parsing fails
        return "ecg_data"

def fetch_and_process_ecg(user_id, start, end, cursor):
    """Fetches data using the dynamically generated partition name."""
    
    table_name = get_partition_name(start)
    
    query = f"""
        SELECT COALESCE(channel1,0), COALESCE(channel2,0), COALESCE(channel3,0)
        FROM {table_name} 
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

# --- STARTUP EVENT (for caching model files) ---
@app.on_event("startup")
async def startup_event():
    print("🚀 Startup: Ensuring model weights are cached...")
    try:
        # Pre-cache main files using the token
        hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILES["config"], token=HF_TOKEN)
        hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILES["classifier"], token=HF_TOKEN)
        hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILES["reconstructor"], token=HF_TOKEN)
        print("✅ Models cached successfully.")
    except Exception as e:
        print(f"❌ Startup Warning: Could not cache models. Will retry on first request. Error: {e}")

# --- ENDPOINTS ---
class AnalysisRequest(BaseModel):
    user_id: str
    start: str
    end: str

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
def analyze_data(request: AnalysisRequest):
    try:
        clear_memory()
        
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                signal = fetch_and_process_ecg(request.user_id, request.start, request.end, cursor)
                if signal is None: return {"accepted": False, "error": "No data"}

                # Features
                lead_ii = signal[1, :]
                peaks = detect_r_peaks(lead_ii, fs=TARGET_FS)
                hrv = compute_hrv_features(peaks, fs=TARGET_FS)

                # Inference
                clf_model = load_classifier()
                
                x = torch.from_numpy(signal).unsqueeze(0).to(device)
                h = torch.from_numpy(hrv).unsqueeze(0).float().to(device)
                a = torch.tensor([0.5], device=device).float()
                s = torch.tensor([0.5], device=device).float()

                with torch.no_grad():
                    logits = clf_model(x, hrv_features=h, age=a, sex=s)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # UNLOAD MODEL IMMEDIATELY TO SAVE RAM
                del clf_model
                clear_memory() 

                top_idx = int(np.argmax(probs))
                pred = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
                conf = float(probs[top_idx])
                all_probs = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}

                # Audit Log
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

                recon_model = load_reconstructor()
                
                x = torch.from_numpy(signal).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = recon_model(x)
                recon_data = out.cpu().numpy()[0].tolist()
                
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