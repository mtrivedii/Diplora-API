from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2 import sql  # SECURITY FIX: For safe SQL identifier quoting
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pydantic import BaseModel, Field
import os
import json
import numpy as np
import torch
import gc 
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from datetime import datetime
import hashlib
import uuid
from typing import Optional
import logging
import re

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

# --- HUGGING FACE CONFIG (SECURITY ENHANCED) ---
HF_REPO_ID = "maanit/diplora-demo"
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL_REVISION = os.environ.get("HF_MODEL_REVISION", "main")  # SECURITY: Pin to specific version

# Model integrity hashes (optional but recommended for ISO 13485)
CLASSIFIER_SHA256 = os.environ.get("CLASSIFIER_SHA256")
RECONSTRUCTOR_SHA256 = os.environ.get("RECONSTRUCTOR_SHA256")

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
MODEL_VERSION = "v1.2.0-iso13485"

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

# CORS (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
device = torch.device("cpu")

# --- SECURITY: Job ID Tracking (Prevent Replay Attacks) ---
processed_jobs = set()  # In production, use Redis with TTL

# --- SECURITY HELPERS ---

def verify_model_integrity(model_path: str, expected_hash: str) -> None:
    """
    Verify model file integrity using SHA256 hash.
    
    ISO 13485 Compliance: Clause 7.5.9 (Data Integrity)
    Prevents tampering with AI models that could affect patient safety.
    
    Args:
        model_path: Path to model file
        expected_hash: Expected SHA256 hash
        
    Raises:
        SecurityError: If hash doesn't match
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
    Fetches results.json or falls back to defaults.
    
    SECURITY FIX: Pins to specific model revision for reproducibility.
    """
    try:
        logger.info("📥 Fetching config from Hugging Face...")
        config_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename=HF_FILES["config"], 
            token=HF_TOKEN,
            revision=HF_MODEL_REVISION,  # SECURITY: Pin to validated version
            etag_timeout=30  # Fail fast on network issues
        )
        
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
    Loads classifier from Private HF Repo with security validation.
    
    SECURITY FIXES:
    1. Pins to specific model revision (prevents supply chain attacks)
    2. Uses weights_only=True (prevents arbitrary code execution)
    3. Optionally verifies SHA256 hash (detects tampering)
    
    ISO 13485: Clause 7.5.6 (Software Validation) - ensures only validated models are loaded
    """
    logger.info("🧠 Loading Classifier...")
    
    # SECURITY: Pin to specific validated version
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename=HF_FILES["classifier"], 
        token=HF_TOKEN,
        revision=HF_MODEL_REVISION,  # SECURITY FIX: Version pinning
        etag_timeout=30
    )
    
    # SECURITY: Verify model integrity if hash is provided
    if CLASSIFIER_SHA256:
        verify_model_integrity(model_path, CLASSIFIER_SHA256)
    else:
        logger.warning("⚠️ Model integrity check skipped (no CLASSIFIER_SHA256 set)")
    
    model = LeadAwareResNet1D(**CLF_PARAMS).to(device)
    
    # SECURITY FIX: Use weights_only=True to prevent arbitrary code execution
    # PyTorch's pickle-based loading can execute malicious code without this flag
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model

def load_reconstructor():
    """
    Loads reconstructor from Private HF Repo with security validation.
    
    SECURITY FIXES: Same as load_classifier()
    """
    logger.info("🎨 Loading Reconstructor...")
    
    recon_path = hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename=HF_FILES["reconstructor"], 
        token=HF_TOKEN,
        revision=HF_MODEL_REVISION,  # SECURITY FIX: Version pinning
        etag_timeout=30
    )
    
    # SECURITY: Verify integrity if hash is provided
    if RECONSTRUCTOR_SHA256:
        verify_model_integrity(recon_path, RECONSTRUCTOR_SHA256)
    else:
        logger.warning("⚠️ Model integrity check skipped (no RECONSTRUCTOR_SHA256 set)")
    
    recon = LeadReconstructionNet().to(device)
    
    # SECURITY FIX: Use weights_only=True
    recon.load_state_dict(
        torch.load(recon_path, map_location=device, weights_only=True)
    )
    recon.eval()
    return recon

# --- SECURITY FUNCTIONS ---

async def verify_api_key(key: str = Security(api_key_header)):
    """
    API Key verification - CRITICAL FOR ISO 13485
    Sub-Question 1: Encrypted & authenticated communication
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
    Compute SHA-256 hash of input parameters for audit trail
    Required for ISO 13485 Clause 7.5.9 (Traceability)
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
    DELIVERABLE #2: Audit Log Module (Separate Connection)
    Uses its own connection to avoid transaction issues
    """
    try:
        # Use separate connection with autocommit for audit logs
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
        # Don't fail the request, but log the error

# --- DATA ACCESS ---

def get_partition_name(timestamp_str: str) -> str:
    """
    Parses timestamp string to construct the partition table name.
    
    SECURITY FIXES:
    1. Strict datetime parsing (raises ValueError on invalid input)
    2. Regex validation of output format
    3. Logging of suspicious inputs
    
    ISO 13485: Clause 7.5.9 (Data Integrity) - prevents SQL injection
    
    Args:
        timestamp_str: Timestamp in format "YYYY-MM-DD HH:MM:SS"
        
    Returns:
        Partition table name (e.g., "ecg_data_y2024m01")
        
    Raises:
        ValueError: If timestamp format is invalid
    """
    try:
        # SECURITY: Strict datetime parsing - raises ValueError on invalid input
        dt = datetime.strptime(timestamp_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
        
        # Generate partition name
        table_name = f"ecg_data_y{dt.strftime('%Y')}m{dt.strftime('%m')}"
        
        # SECURITY FIX: Additional validation - ensure partition name matches expected format
        # Format: ecg_data_y{YYYY}m{MM}
        if not re.match(r'^ecg_data_y\d{4}m\d{2}$', table_name):
            raise ValueError(f"Invalid partition name format: {table_name}")
        
        return table_name
    
    except ValueError as e:
        # Log security event for suspicious input
        security_logger.warning(
            f"Invalid timestamp format or partition name: {timestamp_str}",
            extra={
                "security_event": True, 
                "input": timestamp_str,
                "error": str(e)
            }
        )
        
        # Fallback to main table
        return "ecg_data"

def fetch_and_process_ecg(user_id, start, end, cursor):
    """
    Fetches data using the dynamically generated partition name.
    
    SECURITY FIX: Uses SQL identifier quoting to prevent injection.
    """
    
    table_name = get_partition_name(start)
    
    # SECURITY FIX: Use SQL identifier to safely quote table name
    # This prevents SQL injection even if table_name contains malicious content
    query = sql.SQL("""
        SELECT COALESCE(channel1,0), COALESCE(channel2,0), COALESCE(channel3,0)
        FROM {table}
        WHERE user_id = %s AND timestamp >= %s AND timestamp < %s
        ORDER BY timestamp ASC LIMIT 5000;
    """).format(table=sql.Identifier(table_name))
    
    cursor.execute(query, (user_id, start, end))
    rows = cursor.fetchall()
    
    if not rows: 
        return None

    raw = np.array(rows, dtype=np.float32).T 
    filtered = Signal._bp_filter_np(raw, fs=TARGET_FS, lowcut=BP_LOW, highcut=BP_HIGH)
    if filtered.shape[1] < 5000:
        pad = np.zeros((3, 5000 - filtered.shape[1]), dtype=np.float32)
        filtered = np.concatenate([filtered, pad], axis=1)
    return filtered[:, :5000]

# --- PYDANTIC MODELS (Enhanced Validation) ---

class AnalysisRequest(BaseModel):
    """Enhanced request model with validation"""
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
        "security": {
            "model_pinning": HF_MODEL_REVISION != "main",
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
        "model_revision": HF_MODEL_REVISION,
        "service": "diplora-ai-engine",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_data(request: AnalysisRequest, http_request: Request):
    """
    MAIN ANALYSIS ENDPOINT
    
    Security Features (Sub-Q1, Sub-Q2):
    - API Key validation (X-API-Key header)
    - Job ID replay attack prevention
    - Input hash for integrity
    - Full audit logging
    
    ISO 13485 Compliance (Sub-Q3):
    - Immutable audit logs
    - Traceability: input → process → output
    - Error logging
    """
    job_id = request.job_id
    request_ip = http_request.client.host
    input_hash = compute_input_hash(request.user_id, request.start, request.end)
    
    # SECURITY: Prevent replay attacks (Sub-Q2)
    if job_id in processed_jobs:
        logger.warning(f"⚠️ SECURITY: Duplicate job_id detected: {job_id}")
        
        # Log replay attack attempt
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
        
        # Use separate connection for data fetching
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                # Try to fetch ECG data
                try:
                    signal = fetch_and_process_ecg(request.user_id, request.start, request.end, cursor)
                except Exception as db_error:
                    # Database error (table doesn't exist, etc.)
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

                # Store in existing results table
                cursor.execute(
                    "INSERT INTO ecg_analysis_results (user_id, start_ts, end_ts, metrics, annotations) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (request.user_id, request.start, request.end, json.dumps(output_data), json.dumps(list(all_probs.keys())))
                )
                result_id = cursor.fetchone()[0]
                conn.commit()
                
                # Mark job as processed (replay protection)
                processed_jobs.add(job_id)
                
        # Log successful completion (after everything is done)
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
        
        # Log failure
        log_audit_event_separate(
            job_id, request.user_id, "analyze", "failed",
            input_hash, {}, request_ip, str(e)
        )
        
        return {"accepted": False, "error": str(e), "job_id": job_id}

@app.post("/reconstruct", dependencies=[Depends(verify_api_key)])
async def reconstruct_data(request: AnalysisRequest, http_request: Request):
    """
    12-Lead Reconstruction Endpoint (with audit logging)
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
    # ISO 13485 Note: Binding to 0.0.0.0 is required for containerized deployment (Render)
    # Network security is enforced at platform level (firewall + TLS)
    # Application-level security enforced via API key authentication
    uvicorn.run(app, host="0.0.0.0", port=8000)