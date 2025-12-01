# Diplora AI Engine (Python/FastAPI)

This microservice is the dedicated computation engine for the Diplora cardiac analysis platform. It hosts the TensorFlow Lite inference model and serves as a secure bridge between the Supabase backend and the AI processing logic.

## 🏗️ Architecture

This service is designed to meet **ISO 13485 traceability requirements** by ensuring all analysis requests are authenticated, processed securely, and logged permanently.

* **Host:** Render (PaaS) for dedicated Python compute power.
* **Trigger:** Called via HTTP POST from Supabase Edge Functions.
* **Data Source:** Fetches **Reconstructed 12-Lead ECG** data securely from Supabase (PostgreSQL).
* **Data Sink:** Writes analysis results and audit logs to the `ecg_analysis_results` table.

## 🚀 Features

* **FastAPI:** High-performance, asynchronous Python web framework.
* **Secure Connection:** Uses `psycopg2` with Session Pooling for IPv4/IPv6 compatibility.
* **Traceability:** Automatically generates a `job_id` and logs input parameters and output probabilities for every request.
* **Validation:** Uses Pydantic models to strictly validate incoming JSON payloads.
* **AI Inference:** Runs a TensorFlow Lite (`.tflite`) model for arrhythmia detection.

## 🛠️ Local Setup

### 1. Prerequisites
* Python 3.12+
* A Supabase Project (Staging/Prod)
* Access to Diplora's Modal environment (for model assets)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/mtrivedii/Diplora-API.git
cd Diplora-API

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate  # Windows
# source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt
