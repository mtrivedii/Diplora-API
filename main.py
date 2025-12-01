from fastapi import FastAPI
import psycopg2
import sys
from pydantic import BaseModel
import os
import json 
import numpy as np

# --- AI IMPORTS ---
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
    print("❌ CRITICAL: DB_PASSWORD environment variable not set.")
    sys.exit(1)

DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# Files you HAVE (based on your uploads)
MODEL_PATH = "ecgnet_with_preprocessing.tflite"
CLASS_JSON = "ecg_labels.json"       # Replaces class_names.pkl
THRESHOLDS_JSON = "best_thresholds.json" # For optimized accuracy

# --- AI HELPER FUNCTIONS ---

def load_model(model_path):
    if tflite is None or not os.path.exists(model_path):
        return None, None, None
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def load_assets(json_path, threshold_path):
    """Loads class labels and thresholds from JSON files."""
    # 1. Load Labels
    if not os.path.exists(json_path):
        return ["Error: ecg_labels.json missing"], None
    with open(json_path, 'r') as f:
        class_names = json.load(f) # Expecting a list ["AF", "NSR", ...]

    # 2. Load Thresholds
    if not os.path.exists(threshold_path):
        print("⚠️ Thresholds missing. Using default 0.5.")
        thresholds = np.array([0.5] * len(class_names))
    else:
        with open(threshold_path, 'r') as f:
            thresholds = np.array(json.load(f))
            
    return class_names, thresholds

def predict(interpreter, input_details, output_details, ecg_data, thresholds):
    interpreter.set_tensor(input_details[0]['index'], ecg_data)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]['index'])
    probs = 1 / (1 + np.exp(-logits)) 
    
    # Apply optimized thresholds
    if thresholds is not None:
        bin_preds = (probs[0] >= thresholds).astype(int)
    else:
        bin_preds = (probs[0] >= 0.5).astype(int)
        
    return probs[0], bin_preds

# --- STARTUP ---
interpreter, input_details, output_details = load_model(MODEL_PATH)
class_names, best_thresholds = load_assets(CLASS_JSON, THRESHOLDS_JSON)

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
    
    if interpreter is None:
        return {"accepted": False, "error": "AI model assets missing."}
    
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:

                # --- 1. FETCH RECONSTRUCTED 12-LEAD DATA ---
                # TODO: Replace placeholders with real names from Kaximu
                ecg_query = """
                    SELECT reconstructed_signal_column_name FROM reconstructed_ecg_data_table_name
                    WHERE user_id = %s AND timestamp >= %s AND timestamp < %s
                    ORDER BY timestamp;
                """
                
                cursor.execute(ecg_query, (request.user_id, request.start, request.end))
                reconstructed_data_row = cursor.fetchone()
                
                if reconstructed_data_row is None:
                    return {"accepted": False, "error": "No reconstructed data found."}

                # --- 2. INFERENCE ---
                reconstructed_signal = np.array(reconstructed_data_row[0], dtype=np.float32)
                preprocessed_ecg = reconstructed_signal.reshape((1, 5000, 12)) 
                
                probs, bin_preds = predict(interpreter, input_details, output_details, preprocessed_ecg, best_thresholds)
                
                # --- 3. FORMAT OUTPUT (Matching Diagram 3) ---
                top_index = np.argmax(probs)
                primary_prediction = class_names[top_index]
                confidence_score = float(probs[top_index])
                
                detailed_probabilities = {}
                for i, name in enumerate(class_names):
                    if bin_preds[i] == 1:
                        detailed_probabilities[name] = float(probs[i])

                # --- 4. SAVE RESULTS ---
                print("Saving traceable results...")
                
                result_payload = {
                    "prediction": primary_prediction,
                    "confidence": confidence_score,
                    "probabilities": detailed_probabilities,
                    "signal_quality": "good" 
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

        return {
            "accepted": True,
            "job_id": str(job_id),
            "prediction": primary_prediction,
            "confidence": confidence_score,
            "probabilities": detailed_probabilities,
            "signal_quality": "good"
        }

    except Exception as e:
        print(f"❌ Operation failed: {e}")
        return {"accepted": False, "error": str(e)}