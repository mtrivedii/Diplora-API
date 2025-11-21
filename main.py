from fastapi import FastAPI
import psycopg2
import sys
from pydantic import BaseModel
import os
import json 
import joblib
import numpy as np

# --- AI IMPORTS ---
# Note: The logic for handling missing files is now simplified as we are waiting for them.
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

# Base URL for the Session Pooler connection
DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# AI Model paths (These files must be copied to the repository root)
MODEL_PATH = "ecgnet_with_preprocessing.tflite"
CLASS_PKL = "class_names.pkl"

# --- AI HELPER FUNCTIONS ---

def load_model(model_path):
    """Loads the TFLite model and allocates tensors."""
    if tflite is None or not os.path.exists(model_path):
        return None, None, None
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def load_classes(pkl_path):
    """Loads the class names from the .pkl file."""
    if not os.path.exists(pkl_path):
        return ["Error: class_names.pkl not found"]
        
    return joblib.load(pkl_path)

def predict(interpreter, input_details, output_details, ecg_data, threshold=0.5):
    """Runs inference on preprocessed ECG data."""
    # NOTE: This assumes ecg_data is already a (1, 5000, 12) numpy array.
    interpreter.set_tensor(input_details[0]['index'], ecg_data)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]['index'])
    probs = 1 / (1 + np.exp(-logits)) 
    preds = (probs >= threshold).astype(int)
    return probs[0], preds[0]

# --- LOAD MODEL ON STARTUP ---
# The server will start, but these will be 'None' until the files are added.
interpreter, input_details, output_details = load_model(MODEL_PATH)
class_names = load_classes(CLASS_PKL)

# --- FASTAPI APP & ENDPOINTS ---
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
    
    # 1. Check for missing model assets (This will fail until files are received)
    if interpreter is None or "Error" in class_names[0]:
        return {"accepted": False, "error": "AI model assets are missing or failed to load. Cannot run inference."}
    
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            
            with conn.cursor() as cursor:

                # --- 2. FETCH RECONSTRUCTED 12-LEAD DATA ---
                # This queries the NEW table (output of the Reconstruction Model)
                # NOTE: Placeholder table and column names will be replaced once Taco responds.
                ecg_query = """
                    SELECT reconstructed_signal_column_name FROM reconstructed_ecg_data_table_name
                    WHERE user_id = %s
                    AND timestamp >= %s
                    AND timestamp < %s
                    ORDER BY timestamp;
                """
                
                cursor.execute(ecg_query, (request.user_id, request.start, request.end))
                reconstructed_data_row = cursor.fetchone()
                
                if reconstructed_data_row is None:
                    return {"accepted": False, "error": "No RECONSTRUCTED data found for this time window in the database."}

                # --- 3. PREPROCESS & RUN AI PREDICTION ---
                # NOTE: The PREPROCESSING is now a simple conversion, as the data is pre-processed.
                reconstructed_signal = np.array(reconstructed_data_row[0], dtype=np.float32)
                preprocessed_ecg = reconstructed_signal.reshape((1, 5000, 12)) 
                
                print("Running REAL AI inference...")
                probs, bin_preds = predict(interpreter, input_details, output_details, preprocessed_ecg)
                
                # Format the results
                predictions = {}
                for i, class_name in enumerate(class_names):
                    if bin_preds[i] == 1:
                        predictions[class_name] = float(probs[i])
                
                # --- 4. SAVE RESULTS (Traceability) ---
                print("Saving results to database...")
                
                metrics_json = json.dumps({"probabilities": predictions})
                annotations_json = json.dumps({"labels": list(predictions.keys())})
                
                insert_query = """
                    INSERT INTO ecg_analysis_results
                    (user_id, start_ts, end_ts, metrics, annotations)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                """
                cursor.execute(insert_query, (
                    request.user_id,
                    request.start,
                    request.end,
                    metrics_json,
                    annotations_json
                ))
                
                result_id = cursor.fetchone()[0]
                print(f"✅ Results saved with ID: {result_id}")
        
        # Return a success response
        return {
            "accepted": True, 
            "job_id": str(result_id),
            "predictions": predictions
        }

    except Exception as e:
        print(f"❌ Operation failed: {e}")
        return {"accepted": False, "error": str(e)}