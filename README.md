# Conversational SHL Assessment Recommender

A high-performance, stateless FastAPI service that acts as a Conversational Agent to recommend SHL assessments from the SHL product catalog.

This application was built for the SHL AI Intern Take-home Assignment and is engineered for strict schema compliance, grounded recommendations, and stateless conversational interactions.

---

## 🛠️ Features

- **Stateless Session Management:** The backend maintains no per-session database state, reconstructs context dynamically from the conversation history supplied in each request, and respects the 8-turn conversation budget.
- **Hybrid Retrieval Pipeline:** Combines keyword-based retrieval with lightweight TF-IDF semantic retrieval to improve recommendation quality without requiring an external vector database.
- **Zero-Hallucination Grounding:** All recommendation names, URLs, and test types are programmatically validated against the preprocessed SHL catalog (`catalog_processed.json`) before being returned.
- **Conversation Behaviours:** Supports clarification, recommendation, refinement, comparison, and refusal as required by the assignment.
- **Rate Limit Resilience:** Built-in retry logic to recover from transient LLM provider failures and rate-limit responses.
- **Diagnostics & Mock Mode:** Supports offline Mock Mode to execute replay test suites without consuming LLM API requests.

---

## 🚀 Local Run and Setup

### 1. Prerequisites

Ensure you have **Python 3.10+** installed.

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Configuration

Copy the template configuration file:

```bash
cp .env.example .env
```

Open `.env` and configure:

- `GROQ_API_KEY`: Your Groq API key.
- `LLM_PROVIDER`: `groq`
- `GROQ_MODEL`: `llama-3.1-8b-instant`
- `MOCK_MODE`: Set to `true` for offline testing or `false` for live LLM mode.

---

### 3. Run the Server

Start the service locally:

```bash
python main.py
```

Or run directly with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Verify the server:

```
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

---

### 4. Run Automated Tests

Execute the replay test suite:

```bash
python run_tests.py
```

---

## 📦 Cloud Deployment Guide

The application can be deployed to **Render**, **Railway**, or any platform supporting Python web services.

1. Create a GitHub repository and push the project.
2. Create a new **Web Service** connected to the repository.
3. Configure:

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

4. Configure the following environment variables:

- `GROQ_API_KEY`: Your Groq API key
- `LLM_PROVIDER`: `groq`
- `GROQ_MODEL`: `llama-3.1-8b-instant`
- `MOCK_MODE`: `false`

---

## 📄 Project Structure

- `main.py` — FastAPI backend containing the conversational agent and API endpoints.
- `catalog_processed.json` — Preprocessed SHL assessment catalog.
- `trace_database.json` — Offline conversation traces used for replay testing.
- `run_tests.py` — Automated replay evaluation script.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Git ignore configuration.
- `README.md` — Project documentation.

---

## 🔧 Technology Stack

- FastAPI
- Pydantic
- Groq Llama-3.1-8B-Instant
- Custom Keyword Retrieval
- TF-IDF Semantic Retrieval
- Uvicorn
- Render

---

## 📡 API Endpoints

### Health Check

```
GET /health
```

Returns:

```json
{
  "status": "ok"
}
```

### Chat

```
POST /chat
```

Accepts the conversation history and returns:

- `reply`
- `recommendations`
- `end_of_conversation`

following the required assignment schema.
