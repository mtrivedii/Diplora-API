from fastapi import FastAPI
import psycopg2
import sys
from pydantic import BaseModel
import os
import json 
import joblib
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
CLASS_JSON = "ecg_labels.json"      # Use JSON, not PKL
THRESHOLDS_JSON = "best_thresholds.json"

# --- AI HELPER FUNCTIONS ---

def load_model(model_path):
    if tflite is None or not os.path.exists(model_path):
        return None, None, None
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def load_classes_and_thresholds(json_path, threshold_path):
    # 1. Load Class Names from JSON
    if not os.path.exists(json_path):
        return ["Error: ecg_labels.json not found"], None
    
    with open(json_path, 'r') as f:
        # The file is a simple list ["AB", "AF", ...]
        class_names = json.load(f)

    # 2. Load Thresholds
    if not os.path.exists(threshold_path):
        thresholds = np.array([0.5] * len(class_names))
    else:
        with open(threshold_path, 'r') as f:
            thresholds = np.array(json.load(f))
            
    return class_names, thresholds

def predict(interpreter, input_details, output_details, ecg_data, thresholds):
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], ecg_data)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]['index'])
    probs = 1 / (1 + np.exp(-logits)) 
    
    # Apply specific threshold for each class
    if thresholds is not None:
        bin_preds = (probs[0] >= thresholds).astype(int)
    else:
        bin_preds = (probs[0] >= 0.5).astype(int)
        
    return probs[0], bin_preds

# --- LOAD MODEL ON STARTUP ---
interpreter, input_details, output_details = load_model(MODEL_PATH)
class_names, best_thresholds = load_classes_and_thresholds(CLASS_JSON, THRESHOLDS_JSON)

# --- FASTAPI APP ---
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
    
    # 1. Check if model is loaded
    if interpreter is None:
        return {"accepted": False, "error": "AI model file (tflite) is missing."}
    
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:

                # --- 2. FETCH RECONSTRUCTED DATA ---
                # Placeholder names: ask Bas for the real ones!
                ecg_query = """
                    SELECT reconstructed_signal_column_name FROM reconstructed_ecg_data_table_name
                    WHERE user_id = %s AND timestamp >= %s AND timestamp < %s
                    ORDER BY timestamp;
                """
                cursor.execute(ecg_query, (request.user_id, request.start, request.end))
                row = cursor.fetchone()
                
                if row is None:
                    return {"accepted": False, "error": "No reconstructed data found."}

                # --- 3. PREDICT ---
                # Convert DB blob/array to numpy
                signal = np.array(row[0], dtype=np.float32)
                input_data = signal.reshape((1, 5000, 12)) 
                
                probs, bin_preds = predict(interpreter, input_details, output_details, input_data, best_thresholds)
                
                predictions = {}
                for i, name in enumerate(class_names):
                    if bin_preds[i] == 1:
                        predictions[name] = float(probs[i])
                
                # --- 4. SAVE ---
                insert_query = """
                    INSERT INTO ecg_analysis_results
                    (user_id, start_ts, end_ts, metrics, annotations)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                """
                cursor.execute(insert_query, (
                    request.user_id, request.start, request.end,
                    json.dumps({"probabilities": predictions}),
                    json.dumps({"labels": list(predictions.keys())})
                ))
                result_id = cursor.fetchone()[0]
        
        return {"accepted": True, "job_id": str(result_id), "predictions": predictions}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"accepted": False, "error": str(e)}
    