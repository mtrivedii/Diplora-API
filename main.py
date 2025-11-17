from fastapi import FastAPI
import psycopg2
import sys
from pydantic import BaseModel
import os # Used to get the secret password

# --- 1. CONFIGURATION ---

# Get the database password from an Environment Variable
# This is a security best practice. We will set this in Render.
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# Exit if the password isn't set
if not DB_PASSWORD:
    print("❌ CRITICAL: DB_PASSWORD environment variable not set.")
    sys.exit(1)

# Build the Session Pooler connection string
DATABASE_URL = f"postgresql://postgres.vcdvtrqrqoegtjmtaulm:{DB_PASSWORD}@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

# Create the main FastAPI application
app = FastAPI()

# --- 2. DEFINE DATA MODELS ---

# This defines the JSON input your API expects.
# It matches the payload from the Edge Function
class AnalysisRequest(BaseModel):
    user_id: str
    start: str
    end: str

# --- 3. CREATE API ENDPOINTS ---

# This is a simple "root" endpoint to check if the server is running
@app.get("/")
def read_root():
    return {"status": "Diplora AI Engine is running"}

# This is your main "/analyze" endpoint
@app.post("/analyze")
def analyze_data(request: AnalysisRequest):
    """
    This endpoint:
    1. Receives a user_id and time window.
    2. Connects to the Supabase DB.
    3. Fetches raw ECG and IMU data.
    4. (Future) Runs the AI model.
    5. (Future) Saves the results.
    """
    
    print(f"Received request for user: {request.user_id}")
    
    try:
        # Connect to the database using the Session Pooler string
        with psycopg2.connect(DATABASE_URL) as conn:
            print("✅ DB Connection successful!")
            
            with conn.cursor() as cursor:
                
                # Prepare the SQL query from the docs
                ecg_query = """
                    SELECT * FROM ecg_data
                    WHERE user_id = %s
                    AND timestamp >= %s
                    AND timestamp < %s;
                """
                
                # Execute the query safely
                cursor.execute(ecg_query, (
                    request.user_id,
                    request.start,
                    request.end
                ))
                ecg_results = cursor.fetchall()
                print(f"Found {len(ecg_results)} ECG rows.")

                # --- TODO: AI ANALYSIS GOES HERE ---
                # 1. Add your IMU data query here
                # 2. Load your AI model
                # 3. Run inference on ecg_results and imu_results
                
                # --- TODO: SAVE RESULTS ---
                # 1. Run an INSERT query to save results
                #    to the ecg_analysis_results table
                
        
        # Return a success response (as shown in docs)
        return {"accepted": True, "job_id": "temp-job-id-123"}

    except Exception as e:
        print(f"❌ Operation failed: {e}")
        # Return an error if something went wrong
        return {"accepted": False, "error": str(e)}