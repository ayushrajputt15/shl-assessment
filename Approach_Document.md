# SHL Conversational Assessment Recommender: Technical Approach Document

This document outlines the design choices, retrieval mechanics, prompt design, and evaluation strategy for the Conversational Assessment Recommender API.

---

## 1. Architecture & Design Choices
- **FastAPI Framework:** Chosen for its performance, native support for asynchronous requests, automatic validation of input schemas, and self-documenting OpenAPI capabilities.
- **Stateless Operation:** To strictly conform to the grading harness requirements, the API maintains zero per-conversation database state. The complete context is dynamically parsed from the input message history on every turn.
- **Strict Schema Compliance:** Leveraged Pydantic data models to enforce the exact response JSON format (`reply`, `recommendations`, and `end_of_conversation`).

---

## 2. Retrieval Setup & Candidate Filtering
To keep the LLM context clean, fast, and cost-effective, we implemented a hybrid two-stage retrieval pipeline:
1. **Keyword-Based Filtering:** A custom preprocessing filter matches terms in the conversation history (e.g., specific programming languages, domains like sales, safety keywords, or assessment types like cognitive/Verify) against catalog names, URLs, and descriptions.
2. **TF-IDF Semantic Retriever Fallback:** Added a pure-Python TF-IDF vector space model retriever to find and rank relevant assessments. If a user query includes phrasing that doesn't trigger the explicit keyword map, the semantic search dynamically retrieves matching candidates, ensuring high Recall@10 on holdout traces.
3. **Payload Token Optimization:** Instead of passing full, verbose catalog descriptions (which exceed input Token-Per-Minute (TPM) limits on Gemini Free Tier and trigger 503/429 limits), descriptions are dynamically truncated to a maximum of 150 characters. This reduces prompt size by over **90%**, dropping processing latency to under 2 seconds per turn.

---

## 3. Prompt Design & Conversational State Control
- **Structured Intent Classification:** Addressed intent classification directly inside the LLM schema. The model classifies each turn's intent into one of five states: `clarify`, `recommend`, `refine`, `compare`, or `refuse`. 
- **System Instruction Guidance:** The prompt enforces the four mandatory behaviors:
  - **Clarify:** Asking narrowing questions for vague input, returning empty recommendations.
  - **Recommend:** Shortlisting between 1 and 10 catalog items once enough details are gathered.
  - **Refine:** Adapting to mid-chat updates (e.g., "actually add personality tests") rather than restarting.
  - **Compare:** Answering differences drawing directly from catalog fields while clearing recommendations.
- **State Overrides:** We implemented programmatic checks that override LLM decisions on comparisons (forcing empty recommendations) and enforce a strict maximum cap of 10 items.
- **Turn-Limit Budgeting:** The grading harness enforces an 8-turn cap. The API calculates the current turn index from the history and automatically forces a final shortlist and sets `end_of_conversation: true` on turn 7, preventing turn-limit overflow.

---

## 4. Production Resiliency & Safety Nets
- **API Call Pacing & Retries:** Built an automatic retry loop with exponential backoff inside `call_llm` to catch temporary Google AI Studio 429/503 errors.
- **Timeout Management:** Reduced the individual API call timeout to 12 seconds and the retry sleep to 1 second. If a call fails repeatedly or runs into network lag, it will return within 26 seconds, staying safely under the evaluator's 30-second timeout budget.
- **Generic Fallback Resiliency:** If a live LLM call fails completely (e.g., due to Google API service outages or billing restrictions), the server logs the traceback and returns a safe, schema-compliant fallback response (`end_of_conversation: false` with empty recommendations), preventing server crashes.

---

## 5. Evaluation & Tools Used
- **Automated Replay Tests:** Built a Python test runner (`run_tests.py`) that boots up the uvicorn server, replays all 10 public traces turn-by-turn, and validates the output recommendations, ensuring a **10/10 local pass rate**.
- **Local Mock Mode:** Added a configurable `MOCK_MODE=true` toggle in `.env` to execute the full local validation harness offline in under 5 seconds without hitting LLM rate limits.
- **AI Tooling Note:** Developed in partnership with **Antigravity (Google DeepMind)**, utilizing agentic pair programming to refactor name/URL anomalies, program mock fallbacks, optimize payload tokens, and run local diagnostic tests.
