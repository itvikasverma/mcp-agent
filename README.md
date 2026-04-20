# Repay MCP — Qdrant + LangGraph RAG Stack

A small RAG stack that ingests documents into Qdrant, exposes FastAPI endpoints, and provides a modern web chat UI with MCP tools. Sessions are in-memory (fast, resets on restart).

## Components

1. `app/app.py`
FastAPI service for Qdrant operations (ingest, search, list, init).

2. `services/crawler.py`
Document loader + chunker that posts to `app/app.py`.

3. `services/mcp_server.py`
FastMCP server that exposes tools (`search_documents`, `add_documents`, `init_vector_db`, etc).

4. `web/web.py`
Web UI (chat with sessions) + WebSocket backend using LangGraph + MCP tools.


## Folder Structure

```
app/            # FastAPI service
core/           # Shared config + helpers
services/       # Crawler + MCP server + agents
web/            # Web UI + websocket backend
data/           # Local data (store, exports)
```

## Prerequisites

- Python 3.10+ (recommended 3.11)
- Qdrant running on `http://localhost:6333`
- Environment variables in `.env`

## Environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
QDRANT_URL=http://localhost
QDRANT_API_KEY=optional_if_needed
```

## Install Dependencies

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn qdrant-client langchain-text-splitters langchain-huggingface langchain-qdrant langchain-core langchain-community pdfplumber python-docx python-pptx openpyxl pandas requests python-dotenv langchain-groq fastmcp httpx
```

## Run Qdrant

If you use Docker:

```
docker run -p 6333:6333 qdrant/qdrant
```


## Start Everything

Run both API and Web UI in one command:

```
python main.py
```

This starts:
- API: `http://127.0.0.1:9000`
- MCP server: `services.mcp_server`
- Web UI: `http://127.0.0.1:8000`
- Crawler: one-shot run

## Start the Services


You can start services either manually or via `main.py`.


1. Start the FastAPI backend (port 9000):

```
uvicorn app.app:app --host 127.0.0.1 --port 9000 --reload
```

2. (Optional) Recreate the Qdrant collection:

```
Invoke-RestMethod http://127.0.0.1:9000/init_qdrant
```

3. Run the web UI (port 8000):

```
uvicorn web.web:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```
http://localhost:8000
```

## Ingest Documents

Put files inside:

```
data/store/confluence
```

Then run:

```
python -m services.crawler
```

The crawler posts chunks to `http://127.0.0.1:9000/add_doc`.

## MCP Server (Optional)

If you want the MCP server standalone:

```
python -m services.mcp_server
```

`web/web.py` already launches `services/mcp_server.py` internally via MCP stdio, so running it separately is optional.

## Notes

- Sessions are stored in memory and reset when the `web/web.py` server restarts.
- If you change `.env`, restart the Python processes.

## Troubleshooting

- `ModuleNotFoundError: dotenv`  
  Install with: `pip install python-dotenv`

- `Connection refused` to Qdrant  
  Make sure Qdrant is running on `localhost:6333`.

- No responses in UI  
  Ensure `app/app.py` (port 9000) and `web/web.py` (port 8000) are both running, and `GROQ_API_KEY` is set.
