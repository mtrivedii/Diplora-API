from fastapi import FastAPI
import psycopg2
import sys
from pydantic import BaseModel
import os
import json 
import numpy as np

# --- AI IMPORTS (Safe Import) ---
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    try:
        import tensorflow as tf
        tflite = tf.lite
    except Exception:
        tflite = None

# --- CONFIGURATION ---
DB_PASSWORD = os.environ.get("DB_PASSWORD")
if not DB_PASSWORD:
    # Fallback for local testing if env var is missing
    print("⚠️  DB_PASSWORD not set. Using placeholder...") 
    DB_PASSWORD = "YOUR_DB_PASSWORD_HERE" 

# UPDATE: Correct Connection String
DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# FILES
MODEL_PATH = "ecgnet_with_preprocessing.tflite"
CLASS_JSON = "ecg_labels.json"
THRESHOLDS_JSON = "best_thresholds.json"

app = FastAPI()

class AnalysisRequest(BaseModel):
    user_id: str
    start: str
    end: str

def load_assets(json_path):
    if not os.path.exists(json_path):
        return ["Normal", "Arrhythmia", "Other"] # Fallback defaults
    with open(json_path, 'r') as f:
        return json.load(f)

class_names = load_assets(CLASS_JSON)

@app.get("/")
def read_root():
    return {"status": "Diplora AI Engine is running"}

@app.post("/analyze")
def analyze_data(request: AnalysisRequest):
    print(f"Received request for user: {request.user_id}")
    
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:

                # --- 1. FETCH RAW DATA (Updated for your Schema) ---
                # We fetch the raw channels. The database partitions usually handle the date routing.
                # We limit to 5000 points to match the expected model input size.
                query = """
                    SELECT channel1, channel2, channel3, channel4 
                    FROM ecg_data
                    WHERE user_id = %s AND timestamp >= %s AND timestamp < %s
                    ORDER BY timestamp ASC
                    LIMIT 5000;
                """
                
                cursor.execute(query, (request.user_id, request.start, request.end))
                rows = cursor.fetchall()
                
                if not rows:
                    return {"accepted": False, "error": "No ECG data found for this range."}
                
                print(f"✅ Fetched {len(rows)} data points.")

                # --- 2. MOCK RECONSTRUCTION (Crucial Step) ---
                # Your table has 4 channels. The model expects 12.
                # For the internship 'Architecture' demo, we simulate this transformation.
                raw_signal = np.array(rows, dtype=np.float32) # Shape: (N, 4)
                
                # Expand 4 channels to 12 by repeating them (Just for the Mock!)
                # Shape becomes (N, 12)
                reconstructed_signal = np.tile(raw_signal, (1, 3)) 
                
                # Ensure we have exactly 5000 steps (Zero pad if short)
                if reconstructed_signal.shape[0] < 5000:
                    padding = np.zeros((5000 - reconstructed_signal.shape[0], 12))
                    reconstructed_signal = np.vstack([reconstructed_signal, padding])
                
                # Final Shape for Model: (1, 5000, 12)
                model_input = reconstructed_signal.reshape((1, 5000, 12))

                # --- 3. MOCK INFERENCE (Bypassing PyTorch/TFLite issue) ---
                # Since we don't have the real model, we generate a valid "fake" result.
                # This proves the PIPELINE works, which is your goal.
                
                # Simulate a prediction (e.g., 80% chance of Class 0)
                probs = [0.8, 0.1, 0.1] 
                if len(class_names) > 3: probs = [1.0/len(class_names)] * len(class_names)
                
                top_index = np.argmax(probs)
                primary_prediction = class_names[top_index]
                confidence_score = float(probs[top_index])
                
                detailed_probabilities = {name: p for name, p in zip(class_names, probs)}

                # --- 4. SAVE TRACEABLE RESULTS (ISO 13485 Requirement) ---
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
                conn.commit()

        return {
            "accepted": True,
            "job_id": str(job_id),
            "prediction": primary_prediction,
            "confidence": confidence_score,
            "probabilities": detailed_probabilities
        }

    except Exception as e:
        print(f"❌ Operation failed: {e}")
        return {"accepted": False, "error": str(e)}