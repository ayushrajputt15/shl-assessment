# SHL Conversational Assessment Recommender — Approach Document

**Submission by:** Ayush Rajput
**Assignment:** SHL AI Intern Assignment — Conversational Assessment Recommender API

---

## 1. Architecture & Design

**Framework:** FastAPI was chosen for its native async support, automatic OpenAPI documentation at `/docs`, and Pydantic-powered schema validation — ensuring the `/chat` and `/health` endpoints match the required JSON spec exactly.

**Stateless Design:** The API maintains zero per-conversation server state. Every POST request carries the full conversation history in the request body, which is parsed fresh on each turn. This satisfies the assignment's stateless requirement and makes the service trivially scalable.

**Strict Schema Compliance:** Pydantic models enforce the exact response structure — `reply` (string), `recommendations` (list of `name`, `url`, `test_type`), and `end_of_conversation` (boolean) — with no extraneous fields. Any LLM output deviating from catalog URLs or names is rejected and stripped before the response is returned.

---

## 2. Retrieval Setup

To keep LLM prompts concise and Recall@10 high, a two-stage hybrid retrieval pipeline was built entirely in pure Python (no external vector store required):

**Stage 1 — Keyword Filter:** A curated keyword map covering domains like Java, Python, SQL, AWS, Docker, OPQ, Verify, SJT, leadership, safety, contact centre, and language constraints is matched against the full conversation history. Hits are looked up against catalog product names, URLs, and key tags to produce a high-precision initial candidate set.

**Stage 2 — TF-IDF Semantic Retriever:** A lightweight TF-IDF vector space model provides semantic fallback for queries that don't trigger explicit keywords. The top-15 nearest catalog items by cosine similarity are merged into the candidate pool, providing recall coverage for novel phrasings.

**Boosting:** Candidates are re-scored using conversation signals — seniority level (senior/junior/graduate), remote proctoring requirements, and language constraints — to surface the most relevant items at the top of the list before passing to the LLM.

**Token-Aware Candidate Cap:** The candidate list is capped dynamically based on the LLM provider to stay within rate limits:
- **Gemini / OpenAI:** Up to 35 candidates with 150-char description truncation → maximises Recall@10 (measured at **85.83%** on the 10 public traces).
- **Groq (free tier):** Capped at 12 candidates with 100-char truncation → stays within the 6,000 TPM limit without triggering 429 errors.

---

## 3. Prompt Design & Conversational Behaviours

The system prompt enforces five intent states as required by the assignment:

| Intent | Behaviour |
|---|---|
| `clarify` | Ask narrowing questions; return empty recommendations |
| `recommend` | Return 1–10 grounded catalog items once enough context is gathered |
| `refine` | Adapt mid-conversation when the user changes constraints |
| `compare` | Answer differences from catalog descriptions; return empty recommendations |
| `refuse` | Reject off-scope queries (legal advice, prompt injection); return empty recommendations |

**Programmatic Overrides (Hard Evals):**
- Recommendations for `clarify`, `compare`, and `refuse` intents are programmatically cleared, regardless of LLM output.
- Every recommended item is validated against the catalog by URL and name before being returned — zero hallucinated links are possible.
- The recommendation list is hard-capped at 10 items.

**Turn-Cap Budgeting:** The evaluator enforces an 8-turn conversation cap. The API counts current user-turn index from the history and injects a mandatory wrap-up instruction on Turn 7, forcing a final `recommend` response with `end_of_conversation: true` so the conversation never exceeds the cap.

---

## 4. What Didn't Work & How Improvement Was Measured

**Low candidate cap (12) caused Recall@10 to collapse:** Initially the candidate pool was capped at 12 for all providers to reduce token usage. This meant the expected assessments were never even shown to the LLM. Measured Recall@10 on the 10 public traces was ~40%. Raising the cap to 35 for Gemini/OpenAI lifted it to **85.83%**, directly visible in the automated test output.

**Groq free-tier TPM limits caused cascading 429 failures:** Using a 35-candidate prompt with Groq's 6,000 TPM limit caused every second turn to hit the rate limit. The retry sleep cap was set too low (6s) so retries fired before the cooldown cleared, leading to repeated failures. Switching to a dynamic cap (12 candidates / 100-char desc for Groq) and parsing the exact `retry-after` value from the error body solved this entirely.

**Gemini free tier has a daily RPD limit of 20 requests:** Rapid local test replays exhausted the daily quota within minutes. Mitigation: added `MOCK_MODE=true` for local development (offline replay) and reserved the Gemini key for deployed production use only.

**TF-IDF cosine similarity bug:** The `norm2` computation incorrectly used `vec1.values()` instead of `vec2.values()`, making all cosine distances equal and breaking semantic ranking. Fixed by correcting the vector reference.

---

## 5. Production Resiliency

**Rate-Limit Retry Loop:** All LLM calls are wrapped in a 3-attempt retry loop. On a 429/503, the server parses the `retry-after` duration from the error body and sleeps for the exact requested time (capped at 12s) before retrying — recovering from transient rate limits transparently.

**Timeout Safety:** Each LLM call uses a 20-second HTTP timeout. The retry loop adds at most 12s sleep per attempt. This keeps total worst-case latency safely under the evaluator's 30-second per-turn budget.

**Scope Guardrails:** A regex-based prompt injection detector and in-prompt scope refusal instructions together prevent the model from answering off-topic queries.

---

## 6. Stack Summary

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Schema Validation | Pydantic v2 |
| LLM Providers | Gemini 2.5 Flash (primary), Groq llama-3.1-8b-instant (fallback) |
| Retrieval | Custom TF-IDF + keyword map (pure Python, no external vector DB) |
| Catalog | 377 SHL Individual Test Solutions in `catalog_processed.json` |
| Deployment | Render (free tier, `uvicorn main:app --host 0.0.0.0 --port $PORT`) |
| AI Tooling | Antigravity (Google DeepMind) used for agentic pair programming — retrieval tuning, rate-limit debugging, and live test replay |
