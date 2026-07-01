# Conversational SHL Assessment Recommender

A high-performance, stateless FastAPI service that acts as a Conversational Agent to recommend SHL assessments from the product catalog.

This application is built for the SHL AI Intern Take-home Assignment, engineered for 100% schema alignment, zero link hallucinations, and turn-limit compliance.

---

## 🛠️ Features
- **Stateless Session Management:** The backend maintains no per-session database state, reconstructs context dynamically from the stateless history array, and respects the 8-turn conversation budget.
- **Zero-Hallucination Grounding:** All recommendation URLs, names, and test types are looked up and verified programmatically against a precompiled JSON database (`catalog_processed.json`).
- **Pervasive Rate Limit Resilience:** Built-in automatic retries with exponential backoff on `429 Rate Limit Exceeded` responses from LLM providers.
- **Diagnostics & Mock Mode:** Features a local offline Mock Mode to run replay test suites instantly under 1 second without hitting LLM API rate limits.

---

## 🚀 Local Run and Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Copy the template configuration file:
```bash
cp .env.example .env
```
Open `.env` and fill in your details:
- `GEMINI_API_KEY`: Your Google Gemini API key.
- `LLM_PROVIDER`: `gemini` (default) or `openai`.
- `MOCK_MODE`: Set to `true` for offline testing, or `false` for live LLM mode.

### 3. Run the Server
Start the service locally:
```bash
python main.py
```
Or via Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Visit `http://127.0.0.1:8000/health` in your browser to verify readiness (`{"status": "ok"}`).

### 4. Run Automated Tests
To run the automated replay test suite against the local server:
```bash
python run_tests.py
```

---

## 📦 Cloud Deployment Guide

You can deploy this service to [Render](https://render.com), [Railway](https://railway.app), or any platform supporting Python web services.

1. Create a GitHub repository and push your project files (the `.gitignore` file will ensure your private `.env` and logs are not committed).
2. Set up a **Web Service** linking to the repository.
3. Configure the following deployment parameters:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py` or `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Set the **Environment Variables** in the hosting dashboard:
   - `GEMINI_API_KEY`: `[Your Live API Key]`
   - `LLM_PROVIDER`: `gemini`
   - `MOCK_MODE`: `false` (forces the service to call your live LLM during grading)

---

## 📄 File Structures
- `main.py` - FastAPI backend application containing the core agent logic and REST client.
- `catalog_processed.json` - Processed product catalog.
- `trace_database.json` - Precompiled offline conversation traces database.
- `run_tests.py` - Replay test script.
- `requirements.txt` - Dependency file.
- `.gitignore` - Git ignore rules.
- `Procfile` - Process type declaration for cloud host configurations.
