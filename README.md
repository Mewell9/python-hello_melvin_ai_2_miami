# Coffee Shop Inventory Agent

Natural-language inventory management for a coffee shop supply store. A FastAPI REST API stores products in `products.csv`, and a plain-Python AI agent talks to an LLM (Groq) and calls the API as tools.

**Important:** Do not use agent frameworks (LangChain, LlamaIndex, AutoGen, etc.). The agent loop is implemented manually in `agent.py`.

## Setup

1. Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Open `.env` at the project root and set your Groq API key:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
API_BASE_URL=http://127.0.0.1:8000
```

`.env` is gitignored — never commit your key.

## Running the system

You need **two terminals**. Start the API first.

### Terminal 1 — API

```bash
uvicorn api.app:app --reload
```

The API listens on `http://127.0.0.1:8000`.

### Terminal 2 — Agent

```bash
python agent.py
```

Type natural language (e.g. "We just received 30 liters of oat milk.") and press Enter. Type `quit` or `exit` to stop.

### Stopping

Press `Ctrl + C` in each terminal. Stop the **agent first** so conversation logs finish writing cleanly, then stop the API.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/inventory` | List all products |
| POST | `/inventory` | Add a product (`name`, `quantity`, `unit`) |
| PATCH | `/inventory/{product_id}` | Adjust stock by `delta` (+ delivery, − sale) |
| GET | `/inventory/alerts` | Products below threshold (default 10) |

## Files

- `api/app.py` — FastAPI inventory service
- `products.csv` — Persistent product catalog (source of truth for the API)
- `inventory.md` — Same catalog as a readable markdown table (open Preview in the editor)
- `agent.py` — Manual LLM agent + CLI
- `conversation_log.csv` — Append-only session log (`actor`, `message`, `tool_call`, `timestamp`)
