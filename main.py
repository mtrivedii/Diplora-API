from fastapi import FastAPI
import psycopg2
import sys
from pydantic import BaseModel
import os
import json # For saving results
from pathlib import Path

# --- 1. AI IMPORTS (from Bas's code) ---
import numpy as np
import joblib

# Try to import TFLite, fallback to TensorFlow
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    try:
        import tensorflow as tf
        tflite = tf.lite
    except Exception:
        print("❌ CRITICAL: Neither tflite-runtime nor tensorflow is installed.")
        print("Please add 'tensorflow' to your requirements.txt and redeploy.")
        tflite = None

# --- 2. CONFIGURATION ---

# Database (from environment)
DB_PASSWORD = os.environ.get("DB_PASSWORD")
if not DB_PASSWORD:
    print("❌ CRITICAL: DB_PASSWORD environment variable not set.")
    sys.exit(1)

DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# AI Model paths (from Bas's code)
MODEL_PATH = "ecgnet_with_preprocessing.tflite"
CLASS_PKL = "class_names.pkl"

# --- 3. AI HELPER FUNCTIONS (from Bas's code) ---

def load_model(model_path):
    """Loads the TFLite model and allocates tensors."""
    if tflite is None:
        raise RuntimeError("TensorFlow/TFLite is not available.")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ CRITICAL: Model file not found at {model_path}")
        print("Make sure 'ecgnet_with_preprocessing.tflite' is in your repository.")
        return None, None, None
        
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ AI model loaded successfully.")
    return interpreter, input_details, output_details

def load_classes(pkl_path):
    """Loads the class names from the .pkl file."""
    if not os.path.exists(pkl_path):
        print(f"❌ CRITICAL: Class file not found at {pkl_path}")
        print("Make sure 'class_names.pkl' is in your repository.")
        return ["Error: class_names.pkl not found"]
        
    print("✅ Class names loaded successfully.")
    return joblib.load(pkl_path)

def predict(interpreter, input_details, output_details, ecg_data, threshold=0.5):
    """Runs inference on preprocessed ECG data."""
    interpreter.set_tensor(input_details[0]['index'], ecg_data)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]['index'])
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    preds = (probs >= threshold).astype(int)
    return probs[0], preds[0]

def preprocess_from_db(db_rows, target_len=5000):
    """
    Adapts Bas's logic to use data from the database.
    Converts DB rows into a (1, 5000, 12) NumPy array.
    """
    
    # --- TODO: CRITICAL MISMATCH ---
    # Bas's model [af-detection.ipynb] expects a 12-lead ECG.
    # Your database schema [image_96dbe4.png] seems to only have 4 channels.
    # This function will need to be updated.
    # For now, we will extract the first 4 channels.
    
    # Extract channel data (assuming channel1-4 are at indices 2,3,4,5)
    # This creates a (N, 4) array
    try:
        ecg_data = np.array([row[2:6] for row in db_rows], dtype=np.float32)
    except IndexError:
        print("❌ ERROR: Could not parse ECG data from database rows.")
        return None
    
    # Transpose to (4, N) - (channels, samples)
    ecg_data = ecg_data.T
    
    # --- TODO: Pad/extend the 4 channels to 12 channels ---
    # For now, we'll create a dummy 12-channel array by repeating the 4 channels
    if ecg_data.shape[0] == 4:
        print("⚠️ WARNING: 4-channel data found. Repeating to create 12 dummy channels.")
        ecg_data = np.vstack([ecg_data] * 3) # (12, N)
    
    # --- Preprocessing logic from Bas's code ---
    
    # Normalize (Bas's code used global z-score, the training notebook used per-lead)
    # Using per-lead z-score from the training notebook [af-detection.ipynb]
    ecg_data = (ecg_data - ecg_data.mean(axis=1, keepdims=True)) / (ecg_data.std(axis=1, keepdims=True) + 1e-8)

    # Pad or truncate to target_len
    n_leads, n_samples = ecg_data.shape
    if n_samples > target_len:
        ecg_data = ecg_data[:, :target_len]
    else:
        pad = np.zeros((n_leads, target_len - n_samples), dtype=np.float32)
        ecg_data = np.hstack([ecg_data, pad])

    # Model expects (batch, timesteps, channels)
    # Our shape is (12, 5000), so we transpose and add batch dim
    ecg_data = np.expand_dims(ecg_data.T, axis=0) # (1, 5000, 12)
    
    return ecg_data.astype(np.float32) # TFLite expects float32


# --- 4. LOAD MODEL ON STARTUP ---

# Load the AI model and classes once when the server starts
# This is much faster than loading them on every request
interpreter, input_details, output_details = load_model(MODEL_PATH)
class_names = load_classes(CLASS_PKL)

# Create the main FastAPI application
app = FastAPI()

# --- 5. DEFINE DATA MODELS ---

class AnalysisRequest(BaseModel):
    user_id: str
    start: str
    end: str

# --- 6. CREATE API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "Diplora AI Engine is running"}

@app.post("/analyze")
def analyze_data(request: AnalysisRequest):
    
    print(f"Received request for user: {request.user_id}")
    
    if interpreter is None or class_names is None:
        return {"accepted": False, "error": "AI model is not loaded."}
    
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            print("✅ DB Connection successful!")
            
            with conn.cursor() as cursor:
                
                # --- 1. FETCH DATA ---
                # This query must match your partitioned table [image_96d7a8.png]
                ecg_query = """
                    SELECT * FROM ecg_data
                    WHERE user_id = %s
                    AND timestamp >= %s
                    AND timestamp < %s
                    ORDER BY timestamp;
                """
                cursor.execute(ecg_query, (request.user_id, request.start, request.end))
                ecg_results = cursor.fetchall()
                print(f"Found {len(ecg_results)} ECG rows.")
                
                if len(ecg_results) == 0:
                    return {"accepted": False, "error": "No ECG data found for this time window."}
                
                # --- 2. PREPROCESS DATA ---
                print("Preprocessing data...")
                preprocessed_ecg = preprocess_from_db(ecg_results)
                
                if preprocessed_ecg is None:
                    return {"accepted": False, "error": "Failed to preprocess data."}
                
                print(f"Data shape for model: {preprocessed_ecg.shape}")

                # --- 3. RUN AI PREDICTION ---
                print("Running AI inference...")
                probs, bin_preds = predict(interpreter, input_details, output_details, preprocessed_ecg)
                
                # Format the results
                predictions = {}
                for i, class_name in enumerate(class_names):
                    if bin_preds[i] == 1:
                        predictions[class_name] = float(probs[i])
                
                print(f"Predictions: {predictions}")
                
                # --- 4. SAVE RESULTS (for Traceability ) ---
                print("Saving results to database...")
                
                # Create the JSON objects for the results table [image_4754dc.png]
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