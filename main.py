from fastapi import FastAPI
import psycopg2
import sys
from pydantic import BaseModel
import os
import json 
import numpy as np
from dotenv import load_dotenv # Handles local .env files

# --- CONFIGURATION ---

# 1. Load environment variables (for local testing)
# On Render, this does nothing because there is no .env file, which is fine.
load_dotenv()

# 2. Get the Password securely
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# 3. Validation: Stop the server if the password is missing
if not DB_PASSWORD:
    print("❌ CRITICAL: DB_PASSWORD environment variable not set.")
    print("   -> If running locally: Check your .env file.")
    print("   -> If on Render: Check your Dashboard 'Environment' tab.")
    sys.exit(1)

# 4. Construct the Connection String
# We insert the password dynamically into the URL
DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# --- ASSETS ---
MODEL_PATH = "ecgnet_with_preprocessing.tflite"
CLASS_JSON = "ecg_labels.json"
THRESHOLDS_JSON = "best_thresholds.json"

# --- HELPER: Load Labels ---
def load_assets(json_path):
    if not os.path.exists(json_path):
        return ["Normal Sinus Rhythm", "Arrhythmia", "Other"] # Fallback
    with open(json_path, 'r') as f:
        return json.load(f)

class_names = load_assets(CLASS_JSON)

app = FastAPI()

class AnalysisRequest(BaseModel):
    user_id: str
    start: str
    end: str

@app.get("/")
def read_root():
    return {"status": "Diplora AI Engine is running"}

@app.post("/analyze")
def analyze_data(request: AnalysisRequest):
    print(f"Received request for user: {request.user_id}")
    
    try:
        # Connect to Database
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:

                # --- 1. FETCH RAW DATA ---
                # We query 'ecg_data' and handle NULLs with COALESCE
                ecg_query = """
                    SELECT 
                        COALESCE(channel1, 0), 
                        COALESCE(channel2, 0), 
                        COALESCE(channel3, 0), 
                        COALESCE(channel4, 0)
                    FROM ecg_data
                    WHERE user_id = %s AND timestamp >= %s AND timestamp < %s
                    ORDER BY timestamp ASC
                    LIMIT 5000;
                """
                
                cursor.execute(ecg_query, (request.user_id, request.start, request.end))
                rows = cursor.fetchall()
                
                if not rows:
                    return {"accepted": False, "error": "No ECG data found for this range."}
                
                print(f"✅ Fetched {len(rows)} data points.")

                # --- 2. MOCK RECONSTRUCTION & INFERENCE ---
                # We simulate the AI result because the PyTorch model isn't ready.
                # This proves the TRACEABILITY pipeline works (ISO 13485).
                
                # Mock: 95% confidence it is the first class (e.g. Normal)
                probs = [0.95, 0.02, 0.03] 
                if len(class_names) > 3: 
                    # specific handling if you have many classes
                    probs = [1.0/len(class_names)] * len(class_names)
                    probs[0] = 0.9 # Force a clear winner
                
                top_index = np.argmax(probs)
                primary_prediction = class_names[top_index]
                confidence_score = float(probs[top_index])
                
                detailed_probabilities = {name: p for name, p in zip(class_names, probs)}

                # --- 3. SAVE RESULTS (Traceability) ---
                print("Saving traceable results...")
                
                result_payload = {
                    "prediction": primary_prediction,
                    "confidence": confidence_score,
                    "probabilities": detailed_probabilities,
                    "signal_quality": "good (mocked)" 
                }
                
                insert_query = """
                    INSERT INTO ecg_analysis_results
                    (user_id, start_ts, end_ts, metrics, annotations)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                """
                cursor.execute(insert_query, (
                    request.user_id, request.start, request.end,
                    json.dumps(result_payload),
                    json.dumps({"labels": list(detailed_probabilities.keys())})
                ))
                
                job_id = cursor.fetchone()[0]
                conn.commit() # Commits the log to the database

        return {
            "accepted": True,
            "job_id": str(job_id),
            "prediction": primary_prediction,
            "confidence": confidence_score,
            "probabilities": detailed_probabilities,
            "signal_quality": "good (mocked)"
        }

    except Exception as e:
        print(f"❌ Operation failed: {e}")
        return {"accepted": False, "error": str(e)}