from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pydantic import BaseModel, Field
import os
import json
import numpy as np
import torch
import gc 
from dotenv import load_dotenv
from datetime import datetime
import hashlib
from typing import Optional
import logging
from pathlib import Path
from supabase import create_client, Client

# --- IMPORT MODULES ---
from neural_net import LeadAwareResNet1D
from lead_reconstruction import LeadReconstructionNet
from signal_processing import Signal, compute_hrv_features, detect_r_peaks

load_dotenv()

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
security_logger = logging.getLogger("diplora.security")

# --- CONFIGURATION ---
DB_PASSWORD = os.environ.get("DB_PASSWORD")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD or ''}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# --- SUPABASE STORAGE CONFIG ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # Service role key

# Bucket configuration
STORAGE_BUCKET = "ai-models"

# File paths in bucket
CLASSIFIER_STORAGE_PATH = os.environ.get("CLASSIFIER_STORAGE_PATH", "model_weights.pth")
RECONSTRUCTOR_STORAGE_PATH = os.environ.get("RECONSTRUCTOR_STORAGE_PATH", "lead_reconstruction.pth")
CONFIG_STORAGE_PATH = os.environ.get("CONFIG_STORAGE_PATH", "results.json")

# Model integrity hashes (optional for now)
CLASSIFIER_SHA256 = os.environ.get("CLASSIFIER_SHA256")
RECONSTRUCTOR_SHA256 = os.environ.get("RECONSTRUCTOR_SHA256")

# Local cache directory (Render has /tmp available)
CACHE_DIR = Path("/tmp/diplora_models")
CACHE_DIR.mkdir(exist_ok=True)

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
MODEL_VERSION = "v1.4.0-parent-table"

# --- CUSTOM EXCEPTIONS ---
class SecurityError(Exception):
    """Raised when a security validation fails"""
    pass

# --- APP SETUP ---
app = FastAPI(
    title="Diplora AI Analysis Service",
    description="ISO 13485 Compliant AI Inference Engine",
    version=MODEL_VERSION
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
device = torch.device("cpu")

# --- SECURITY: Job ID Tracking ---
processed_jobs = set()

# --- SUPABASE CLIENT (Singleton) ---
_supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    """Get or create Supabase client (singleton pattern)."""
    global _supabase_client
    if _supabase_client is None:
        logger.info("🔑 Initializing Supabase client...")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase client ready")
    return _supabase_client

# --- STORAGE HELPERS ---

def download_from_supabase(storage_path: str, local_filename: str) -> str:
    """
    Download file from Supabase Storage to local cache using SDK.
    
    Args:
        storage_path: Path in Supabase Storage (e.g., "model_weights.pth")
        local_filename: Filename to save locally (e.g., "classifier.pth")
    
    Returns:
        Local file path
        
    Raises:
        RuntimeError: If download fails
    """
    local_path = CACHE_DIR / local_filename
    
    # Check if already cached
    if local_path.exists():
        logger.info(f"✅ Using cached model: {local_path}")
        return str(local_path)
    
    logger.info(f"📥 Downloading from Supabase Storage: {storage_path}")
    
    try:
        supabase = get_supabase_client()
        
        logger.info(f"🔗 Bucket: {STORAGE_BUCKET}, Path: {storage_path}")
        
        # Download file content as bytes
        response = supabase.storage.from_(STORAGE_BUCKET).download(storage_path)
        
        # Save to local cache
        with open(local_path, "wb") as f:
            f.write(response)
        
        logger.info(f"✅ Downloaded to: {local_path} ({len(response):,} bytes)")
        return str(local_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to download {storage_path}: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Bucket: {STORAGE_BUCKET}")
        logger.error(f"   Path: {storage_path}")
        raise RuntimeError(f"Model download failed: {e}")

# --- SECURITY HELPERS ---

def verify_model_integrity(model_path: str, expected_hash: str) -> None:
    """
    Verify model file integrity using SHA256 hash.
    
    ISO 13485 Compliance: Clause 7.5.9 (Data Integrity)
    Prevents tampering with AI models that could affect patient safety.
    """
    actual_hash = hashlib.sha256(open(model_path, 'rb').read()).hexdigest()
    
    if actual_hash != expected_hash:
        security_logger.critical(
            f"MODEL_INTEGRITY_VIOLATION: Expected {expected_hash}, got {actual_hash}",
            extra={
                "event_type": "security_violation",
                "severity": "critical",
                "model_path": model_path
            }
        )
        raise SecurityError(
            f"Model integrity check failed. "
            f"Expected: {expected_hash}, Got: {actual_hash}"
        )
    
    logger.info(f"✅ Model integrity verified: {model_path}")

# --- CORE HELPERS ---

def clear_memory():
    """Forces garbage collection to free RAM."""
    gc.collect()

def load_config():
    """
    Fetches results.json from Supabase Storage or falls back to defaults.
    """
    try:
        logger.info("📥 Fetching config from Supabase Storage...")
        config_path = download_from_supabase(CONFIG_STORAGE_PATH, "results.json")
        
        with open(config_path, "r") as f:
            meta = json.load(f)
            
        logger.info("✅ Config loaded successfully")
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
        logger.warning(f"⚠️ Config download failed ({e}). Using FALLBACK defaults.")
        return {
            "class_names": DEFAULT_CLASSES,
            "params": DEFAULT_ARCH
        }

# Load config immediately
CONFIG_DATA = load_config()
CLASS_NAMES = CONFIG_DATA["class_names"]
CLF_PARAMS = CONFIG_DATA["params"]

def load_classifier():
    """
    Loads classifier from Supabase Storage with security validation.
    """
    logger.info("🧠 Loading Classifier from Supabase Storage...")
    
    # Download from Supabase
    model_path = download_from_supabase(CLASSIFIER_STORAGE_PATH, "classifier.pth")
    
    # SECURITY: Verify model integrity if hash is provided
    if CLASSIFIER_SHA256:
        verify_model_integrity(model_path, CLASSIFIER_SHA256)
    else:
        logger.warning("⚠️ Model integrity check skipped (no CLASSIFIER_SHA256 set)")
    
    model = LeadAwareResNet1D(**CLF_PARAMS).to(device)
    
    # SECURITY: Use weights_only=True to prevent arbitrary code execution
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    logger.info("✅ Classifier loaded successfully")
    return model

def load_reconstructor():
    """
    Loads reconstructor from Supabase Storage with security validation.
    """
    logger.info("🎨 Loading Reconstructor from Supabase Storage...")
    
    # Download from Supabase
    recon_path = download_from_supabase(RECONSTRUCTOR_STORAGE_PATH, "reconstructor.pth")
    
    # SECURITY: Verify integrity if hash is provided
    if RECONSTRUCTOR_SHA256:
        verify_model_integrity(recon_path, RECONSTRUCTOR_SHA256)
    else:
        logger.warning("⚠️ Model integrity check skipped (no RECONSTRUCTOR_SHA256 set)")
    
    recon = LeadReconstructionNet().to(device)
    
    # SECURITY: Use weights_only=True
    recon.load_state_dict(
        torch.load(recon_path, map_location=device, weights_only=True)
    )
    recon.eval()
    logger.info("✅ Reconstructor loaded successfully")
    return recon

# --- SECURITY FUNCTIONS ---

async def verify_api_key(key: str = Security(api_key_header)):
    """
    API Key verification - CRITICAL FOR ISO 13485
    """
    if not API_SECRET_KEY:
        raise HTTPException(status_code=500, detail="API_SECRET_KEY not configured")
    
    if not key:
        raise HTTPException(status_code=403, detail="Missing X-API-Key header")
    
    if key != API_SECRET_KEY:
        logger.warning(f"⚠️ SECURITY: Invalid API key attempt")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    return True

def compute_input_hash(user_id: str, start: str, end: str) -> str:
    """
    Compute SHA-256 hash of input parameters for audit trail.
    Required for ISO 13485 Clause 7.5.9 (Traceability).
    """
    input_str = f"{user_id}|{start}|{end}"
    return hashlib.sha256(input_str.encode()).hexdigest()

def log_audit_event_separate(
    job_id: str,
    user_id: str,
    action_type: str,
    status: str,
    input_hash: str,
    output_data: dict,
    request_ip: str,
    error_message: Optional[str] = None
):
    """
    Audit Log Module - Uses separate connection to avoid transaction issues.
    ISO 13485 Compliance: Clause 7.5.9 (Traceability).
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_logs (
                job_id, user_id, action_type, status, 
                input_data_hash, output_data, request_ip, 
                error_message, model_version, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            job_id, user_id, action_type, status,
            input_hash, json.dumps(output_data), request_ip,
            error_message, MODEL_VERSION
        ))
        
        cursor.close()
        conn.close()
        logger.info(f"✅ Audit log recorded: {job_id} - {action_type} - {status}")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Audit logging failed: {e}")

# --- DATA ACCESS ---

def fetch_and_process_ecg(user_id, start, end, cursor):
    """
    Fetches ECG data from the partitioned ecg_data table.
    
    PostgreSQL automatically routes queries to the correct partition
    based on the timestamp range.
    
    FIXED for ISO 13485: Query uses Postgres INTERVAL to guarantee
    exactly 10 seconds of data from the start timestamp, eliminating
    client-side timezone bugs.
    """
    query = """
        SELECT COALESCE(channel1, 0), COALESCE(channel2, 0), COALESCE(channel3, 0)
        FROM ecg_data
        WHERE user_id = %s 
          AND timestamp >= %s 
          AND timestamp < %s::timestamptz + INTERVAL '10 seconds'
        ORDER BY timestamp ASC 
        LIMIT 5000;
    """
    
    # We pass the 'start' timestamp twice for both bounds
    cursor.execute(query, (user_id, start, start))
    rows = cursor.fetchall()
    
    if not rows:
        logger.warning(f"No ECG data found for user {user_id} starting at {start}")
        return None
    
    logger.info(f"📊 Fetched {len(rows)} ECG samples for user {user_id}")

    # Convert to numpy array [3, N] where N is number of samples
    raw = np.array(rows, dtype=np.float32).T
    
    # Apply bandpass filter
    filtered = Signal._bp_filter_np(raw, fs=TARGET_FS, lowcut=BP_LOW, highcut=BP_HIGH)
    
    # Pad to 5000 samples if needed (10 seconds at 500 Hz)
    if filtered.shape[1] < 5000:
        pad = np.zeros((3, 5000 - filtered.shape[1]), dtype=np.float32)
        filtered = np.concatenate([filtered, pad], axis=1)
    
    return filtered[:, :5000]

# --- PYDANTIC MODELS ---

class AnalysisRequest(BaseModel):
    """Request model with validation"""
    job_id: str = Field(..., description="Unique job identifier from Edge Function")
    user_id: str = Field(..., min_length=1, max_length=100)
    start: str = Field(..., description="Start timestamp (ISO format)")
    end: str = Field(..., description="End timestamp (ISO format)")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "patient_123",
                "start": "2024-01-15 10:30:00",
                "end": "2024-01-15 10:30:10"
            }
        }

# --- ENDPOINTS ---

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Diplora AI Engine",
        "version": MODEL_VERSION,
        "status": "online",
        "storage": {
            "provider": "Supabase Storage (Python SDK)",
            "bucket": STORAGE_BUCKET,
            "classifier": CLASSIFIER_STORAGE_PATH,
            "reconstructor": RECONSTRUCTOR_STORAGE_PATH,
            "config": CONFIG_STORAGE_PATH
        },
        "database": {
            "ecg_table": "ecg_data (partitioned)",
            "query_mode": "parent table (auto-routing)"
        },
        "security": {
            "model_pinning": True,
            "integrity_checks": bool(CLASSIFIER_SHA256 and RECONSTRUCTOR_SHA256)
        },
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST, requires X-API-Key)",
            "reconstruct": "/reconstruct (POST, requires X-API-Key)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "storage_provider": "Supabase Storage",
        "storage_bucket": STORAGE_BUCKET,
        "service": "diplora-ai-engine",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_data(request: AnalysisRequest, http_request: Request):
    """
    Main ECG Analysis Endpoint
    
    Security Features:
    - API Key validation (X-API-Key header)
    - Job ID replay attack prevention
    - Input hash for integrity
    - Full audit logging
    
    ISO 13485 Compliance:
    - Immutable audit logs
    - Traceability: input → process → output
    - Error logging
    """
    job_id = request.job_id
    request_ip = http_request.client.host
    input_hash = compute_input_hash(request.user_id, request.start, request.end)
    
    # SECURITY: Prevent replay attacks
    if job_id in processed_jobs:
        logger.warning(f"⚠️ SECURITY: Duplicate job_id detected: {job_id}")
        
        log_audit_event_separate(
            job_id, request.user_id, "analyze", "failed",
            input_hash, {"reason": "replay_attack"}, request_ip, 
            "Duplicate job_id (replay attack prevented)"
        )
        
        return {
            "accepted": False, 
            "error": "Duplicate job_id (replay attack prevented)",
            "job_id": job_id
        }
    
    # Log trigger event (before processing)
    log_audit_event_separate(
        job_id, request.user_id, "trigger", "started",
        input_hash, {}, request_ip
    )
    
    try:
        clear_memory()
        
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                # Fetch ECG data using reliable INTERVAL query
                try:
                    signal = fetch_and_process_ecg(request.user_id, request.start, request.end, cursor)
                except Exception as db_error:
                    error_msg = str(db_error)
                    
                    log_audit_event_separate(
                        job_id, request.user_id, "analyze", "failed",
                        input_hash, {"error_type": "database"}, request_ip, error_msg
                    )
                    
                    return {"accepted": False, "error": f"Database error: {error_msg}", "job_id": job_id}
                
                if signal is None:
                    log_audit_event_separate(
                        job_id, request.user_id, "analyze", "failed",
                        input_hash, {"error_type": "no_data"}, request_ip, "No ECG data found"
                    )
                    return {"accepted": False, "error": "No data", "job_id": job_id}

                # Extract HRV features from Lead II
                lead_ii = signal[1, :]
                peaks = detect_r_peaks(lead_ii, fs=TARGET_FS)
                hrv = compute_hrv_features(peaks, fs=TARGET_FS)

                # Load model and run inference
                clf_model = load_classifier()
                
                x = torch.from_numpy(signal).unsqueeze(0).to(device)
                h = torch.from_numpy(hrv).unsqueeze(0).float().to(device)
                a = torch.tensor([0.5], device=device).float()
                s = torch.tensor([0.5], device=device).float()

                with torch.no_grad():
                    logits = clf_model(x, hrv_features=h, age=a, sex=s)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                del clf_model
                clear_memory() 

                top_idx = int(np.argmax(probs))
                pred = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
                conf = float(probs[top_idx])
                all_probs = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}

                # Prepare output
                output_data = {
                    "type": "classification",
                    "prediction": pred,
                    "confidence": conf,
                    "probabilities": all_probs,
                    "hrv_detected": bool(len(peaks) > 0),
                    "r_peaks_count": int(len(peaks))
                }

                # Store result
                cursor.execute(
                    "INSERT INTO ecg_analysis_results (user_id, start_ts, end_ts, metrics, annotations) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (request.user_id, request.start, request.end, json.dumps(output_data), json.dumps(list(all_probs.keys())))
                )
                result_id = cursor.fetchone()[0]
                conn.commit()
                
                # Mark job as processed (replay protection)
                processed_jobs.add(job_id)
                
        # Log successful completion
        log_audit_event_separate(
            job_id, request.user_id, "analyze", "completed",
            input_hash, output_data, request_ip
        )
        
        return {
            "accepted": True,
            "job_id": job_id,
            "result_id": result_id,
            "prediction": pred,
            "confidence": conf,
            "model_version": MODEL_VERSION
        }

    except Exception as e:
        clear_memory()
        logger.error(f"❌ Error in /analyze: {e}")
        
        log_audit_event_separate(
            job_id, request.user_id, "analyze", "failed",
            input_hash, {}, request_ip, str(e)
        )
        
        return {"accepted": False, "error": str(e), "job_id": job_id}

@app.post("/reconstruct", dependencies=[Depends(verify_api_key)])
async def reconstruct_data(request: AnalysisRequest, http_request: Request):
    """
    12-Lead ECG Reconstruction Endpoint
    
    Reconstructs full 12-lead ECG from 3-lead input using neural network.
    """
    job_id = request.job_id
    request_ip = http_request.client.host
    input_hash = compute_input_hash(request.user_id, request.start, request.end)
    
    if job_id in processed_jobs:
        log_audit_event_separate(
            job_id, request.user_id, "reconstruct", "failed",
            input_hash, {"reason": "replay_attack"}, request_ip,
            "Duplicate job_id"
        )
        return {"accepted": False, "error": "Duplicate job_id", "job_id": job_id}
    
    log_audit_event_separate(
        job_id, request.user_id, "reconstruct", "started",
        input_hash, {}, request_ip
    )
    
    try:
        clear_memory()
        
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                try:
                    signal = fetch_and_process_ecg(request.user_id, request.start, request.end, cursor)
                except Exception as db_error:
                    error_msg = str(db_error)
                    log_audit_event_separate(
                        job_id, request.user_id, "reconstruct", "failed",
                        input_hash, {}, request_ip, error_msg
                    )
                    return {"accepted": False, "error": f"Database error: {error_msg}", "job_id": job_id}
                
                if signal is None:
                    log_audit_event_separate(
                        job_id, request.user_id, "reconstruct", "failed",
                        input_hash, {}, request_ip, "No data"
                    )
                    return {"accepted": False, "error": "No data", "job_id": job_id}

                recon_model = load_reconstructor()
                
                x = torch.from_numpy(signal).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = recon_model(x)
                recon_data = out.cpu().numpy()[0].tolist()
                
                del recon_model
                clear_memory()

                output_data = {
                    "type": "reconstruction",
                    "status": "success",
                    "leads_generated": 12
                }

                cursor.execute(
                    "INSERT INTO ecg_analysis_results (user_id, start_ts, end_ts, metrics) VALUES (%s, %s, %s, %s) RETURNING id",
                    (request.user_id, request.start, request.end, json.dumps(output_data))
                )
                result_id = cursor.fetchone()[0]
                conn.commit()
                
                processed_jobs.add(job_id)

        log_audit_event_separate(
            job_id, request.user_id, "reconstruct", "completed",
            input_hash, output_data, request_ip
        )

        return {
            "accepted": True,
            "job_id": job_id,
            "result_id": result_id,
            "reconstruction": recon_data
        }

    except Exception as e:
        clear_memory()
        log_audit_event_separate(
            job_id, request.user_id, "reconstruct", "failed",
            input_hash, {}, request_ip, str(e)
        )
        
        return {"accepted": False, "error": str(e), "job_id": job_id}

if __name__ == "__main__":
    import uvicorn
    # Note: Binding to 0.0.0.0 is required for containerized deployment
    uvicorn.run(app, host="0.0.0.0", port=8000)  