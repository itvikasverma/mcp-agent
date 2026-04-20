import os
import sys
import asyncio
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from typing import Annotated, TypedDict, Dict, List, Optional

load_dotenv()

# Fix Windows asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ───────────── State Definition ───────────── #
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ───────────── FastAPI Setup ───────────── #
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_agent()
    yield

app = FastAPI(lifespan=lifespan)
# app.mount("/static", StaticFiles(directory="static"), name="static")  # For frontend files

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ In-memory Session Store â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ #
sessions: Dict[str, Dict] = {}

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def create_session(title: Optional[str] = None) -> Dict:
    session_id = uuid.uuid4().hex
    session = {
        "id": session_id,
        "title": title or "New chat",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "log": [],
        "lc_messages": [],
    }
    sessions[session_id] = session
    return session

def append_message(session: Dict, role: str, content: str) -> None:
    session["log"].append({
        "role": role,
        "content": content,
        "ts": _now_iso(),
    })
    session["updated_at"] = _now_iso()
    if role == "user":
        session["lc_messages"].append(HumanMessage(content=content))
    elif role == "assistant":
        session["lc_messages"].append(AIMessage(content=content))

# ───────────── Load MCP tools & LLM ───────────── #
client = None
tools = None
model = None
workflow_app = None

async def init_agent():
    global client, tools, model, workflow_app
    client = MultiServerMCPClient({
        "qdrant_tools": {
            "command": sys.executable,
            "args": ["-m", "services.mcp_server"],
            "transport": "stdio"
        }
    })
    tools = await client.get_tools()
    system_prompt = SystemMessage(content=(
        "You are a helpful Research Assistant with access to documents. "
        "Use 'search_documents' tool only for document queries."
    ))

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    ).bind_tools(tools)

    workflow = StateGraph(State)
    def call_model(state: State):
        response = model.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    workflow.add_node("agent", call_model)
    workflow.add_node("action", ToolNode(tools))
    workflow.add_edge(START, "agent")

    def should_continue(state: State):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "action"
        return END

    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("action", "agent")

    workflow_app = workflow.compile()


# ----- Sessions API ----- #
@app.post("/api/sessions")
async def api_create_session():
    session = create_session()
    return {
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "messages": [],
    }

@app.get("/api/sessions")
async def api_list_sessions():
    def preview(log):
        if not log:
            return ""
        last = log[-1]["content"]
        return last[:120]

    items = []
    for session in sessions.values():
        items.append({
            "id": session["id"],
            "title": session["title"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "count": len(session["log"]),
            "preview": preview(session["log"]),
        })
    items.sort(key=lambda s: s["updated_at"], reverse=True)
    return {"sessions": items}

@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "messages": session["log"],
    }

@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"status": "deleted"}


# ───────────── WebSocket Endpoint ───────────── #
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id")
    if not session_id or session_id not in sessions:
        session = create_session()
        session_id = session["id"]
    session = sessions[session_id]
    chat_history = session["lc_messages"]

    try:
        while True:
            data = await websocket.receive_text()
            append_message(session, "user", data)
            final_text = ""

            async for event in workflow_app.astream({"messages": chat_history}, stream_mode="values"):
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content and not last_msg.tool_calls:
                        final_text = last_msg.content

                    # Stream partial responses
                    await websocket.send_text(final_text)

            append_message(session, "assistant", final_text)
    except WebSocketDisconnect:
        print("Client disconnected")

# ───────────── Simple HTML Frontend ───────────── #
@app.get("/")
async def get():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Atlas Chat</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@300;400;500;600;700&family=Fraunces:wght@400;600;700&display=swap');

            :root {
                --ink: #121018;
                --subtle: #6f6877;
                --paper: #f8f2ea;
                --sand: #efe4d6;
                --accent: #2b6ae3;
                --accent-dark: #1e4aa3;
                --glow: #e9d7c3;
                --card: #ffffff;
                --stroke: #e0d4c6;
            }

            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: 'Archivo', sans-serif;
                color: var(--ink);
                background: radial-gradient(circle at top left, #fff5e8 0%, #f6efe7 55%, #f0e4d6 100%);
                min-height: 100vh;
                height: 100vh;
                overflow: hidden;
            }

            .app {
                display: grid;
                grid-template-columns: 280px 1fr;
                height: 100vh;
            }

            .sidebar {
                padding: 24px;
                background: linear-gradient(160deg, #fffaf3 0%, #f4e8d9 100%);
                border-right: 1px solid var(--stroke);
                display: flex;
                flex-direction: column;
                gap: 18px;
                height: 100vh;
            }

            .brand {
                display: flex;
                flex-direction: column;
                gap: 6px;
            }

            .brand h1 {
                font-family: 'Fraunces', serif;
                font-size: 24px;
                margin: 0;
                letter-spacing: 0.5px;
            }

            .brand p {
                margin: 0;
                font-size: 13px;
                color: var(--subtle);
            }

            .new-chat {
                border: none;
                background: var(--accent);
                color: white;
                padding: 12px 14px;
                border-radius: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                box-shadow: 0 8px 20px rgba(43, 106, 227, 0.25);
            }

            .new-chat:hover { transform: translateY(-1px); }

            .session-list {
                display: flex;
                flex-direction: column;
                gap: 10px;
                overflow-y: auto;
                padding-right: 4px;
                padding-bottom: 12px;
                flex: 1;
            }

            .session-card {
                background: var(--card);
                border: 1px solid var(--stroke);
                border-radius: 14px;
                padding: 12px 12px;
                cursor: pointer;
                transition: border 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
            }

            .session-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 10px;
            }

            .delete-btn {
                border: none;
                background: transparent;
                color: #a45c5c;
                font-size: 16px;
                cursor: pointer;
                padding: 2px 6px;
                border-radius: 8px;
            }

            .delete-btn:hover {
                background: #f8e9e9;
            }

            .session-card.active {
                border: 1px solid var(--accent);
                box-shadow: 0 8px 18px rgba(43, 106, 227, 0.12);
            }

            .session-title {
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 6px;
            }

            .session-preview {
                font-size: 12px;
                color: var(--subtle);
            }

            .main {
                display: flex;
                flex-direction: column;
                padding: 28px 32px 22px;
                height: 100vh;
            }

            .topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 18px;
            }

            .topbar h2 {
                font-family: 'Fraunces', serif;
                font-size: 28px;
                margin: 0;
            }

            .status {
                font-size: 12px;
                color: var(--subtle);
                padding: 6px 12px;
                border-radius: 999px;
                background: #ffffffb5;
                border: 1px solid var(--stroke);
            }

            .chat {
                flex: 1;
                background: var(--card);
                border-radius: 18px;
                border: 1px solid var(--stroke);
                padding: 24px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 14px;
                box-shadow: 0 20px 40px rgba(44, 35, 22, 0.08);
            }

            .bubble {
                max-width: 70%;
                padding: 14px 16px;
                border-radius: 16px;
                line-height: 1.5;
                font-size: 15px;
                white-space: pre-wrap;
            }

            .bubble.user {
                align-self: flex-end;
                background: #f3f6ff;
                border: 1px solid #d9e3ff;
            }

            .bubble.ai {
                align-self: flex-start;
                background: #fff7ee;
                border: 1px solid #f1ddc7;
            }

            .composer {
                margin-top: 18px;
                display: grid;
                grid-template-columns: 1fr auto;
                gap: 12px;
                align-items: center;
            }

            .composer textarea {
                width: 100%;
                border-radius: 14px;
                border: 1px solid var(--stroke);
                padding: 14px 16px;
                font-size: 15px;
                font-family: 'Archivo', sans-serif;
                resize: none;
                min-height: 56px;
                background: #fffdf9;
            }

            .send {
                border: none;
                background: var(--accent);
                color: white;
                padding: 14px 18px;
                border-radius: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                box-shadow: 0 10px 22px rgba(43, 106, 227, 0.25);
            }

            .send:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                box-shadow: none;
            }

            .send:hover:not(:disabled) { transform: translateY(-1px); }

            @media (max-width: 900px) {
                .app { grid-template-columns: 1fr; }
                .sidebar { border-right: none; border-bottom: 1px solid var(--stroke); }
            }
        </style>
    </head>
    <body>
        <div class="app">
            <aside class="sidebar">
                <div class="brand">
                    <h1>Atlas Chat</h1>
                    <p>Session-based research assistant</p>
                </div>
                <button class="new-chat" id="newChat">New session</button>
                <div class="session-list" id="sessionList"></div>
            </aside>

            <main class="main">
                <div class="topbar">
                    <h2 id="sessionTitle">New chat</h2>
                    <div class="status" id="status">Offline</div>
                </div>
                <div class="chat" id="chat"></div>
                <div class="composer">
                    <textarea id="input" placeholder="Ask anything about your docs..."></textarea>
                    <button class="send" id="send" disabled>Send</button>
                </div>
            </main>
        </div>

        <script>
            const sessionList = document.getElementById('sessionList');
            const chat = document.getElementById('chat');
            const input = document.getElementById('input');
            const send = document.getElementById('send');
            const newChat = document.getElementById('newChat');
            const statusEl = document.getElementById('status');
            const sessionTitle = document.getElementById('sessionTitle');

            let sessionId = localStorage.getItem('sessionId');
            let ws = null;
            let streamingBubble = null;

            function setStatus(text) {
                statusEl.textContent = text;
            }

            function renderBubble(role, content) {
                const bubble = document.createElement('div');
                bubble.className = `bubble ${role}`;
                bubble.textContent = content;
                chat.appendChild(bubble);
                chat.scrollTop = chat.scrollHeight;
                return bubble;
            }

            function clearChat() {
                chat.innerHTML = '';
            }

            async function fetchSessions() {
                const res = await fetch('/api/sessions');
                const data = await res.json();
                sessionList.innerHTML = '';
                data.sessions.forEach((s) => {
                    const card = document.createElement('div');
                    card.className = 'session-card' + (s.id === sessionId ? ' active' : '');
                    card.innerHTML = `
                        <div class="session-head">
                            <div class="session-title">${s.title}</div>
                            <button class="delete-btn" data-id="${s.id}" title="Delete">?</button>
                        </div>
                        <div class="session-preview">${s.preview || 'No messages yet'}</div>
                    `;
                    card.onclick = () => loadSession(s.id);
                    card.querySelector('.delete-btn').onclick = async (e) => {
                        e.stopPropagation();
                        await deleteSession(s.id);
                    };
                    sessionList.appendChild(card);
                });
            }

            async function deleteSession(id) {
                await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
                if (id === sessionId) {
                    sessionId = null;
                    localStorage.removeItem('sessionId');
                    await createSession();
                    return;
                }
                await fetchSessions();
            }

            async function createSession() {
                const res = await fetch('/api/sessions', { method: 'POST' });
                const session = await res.json();
                sessionId = session.id;
                localStorage.setItem('sessionId', sessionId);
                sessionTitle.textContent = session.title;
                clearChat();
                await fetchSessions();
                connectWs();
            }

            async function loadSession(id) {
                const res = await fetch(`/api/sessions/${id}`);
                if (!res.ok) {
                    await createSession();
                    return;
                }
                const session = await res.json();
                sessionId = session.id;
                localStorage.setItem('sessionId', sessionId);
                sessionTitle.textContent = session.title;
                clearChat();
                session.messages.forEach((m) => renderBubble(m.role === 'assistant' ? 'ai' : 'user', m.content));
                await fetchSessions();
                connectWs();
            }

            function connectWs() {
                if (!sessionId) return;
                if (ws) ws.close();
                setStatus('Connecting');
                ws = new WebSocket(`ws://${location.host}/ws/chat?session_id=${sessionId}`);

                ws.onopen = () => {
                    setStatus('Online');
                    send.disabled = false;
                };

                ws.onclose = () => {
                    setStatus('Offline');
                    send.disabled = true;
                };

                ws.onmessage = (event) => {
                    if (!streamingBubble) {
                        streamingBubble = renderBubble('ai', '');
                    }
                    streamingBubble.textContent = event.data;
                    chat.scrollTop = chat.scrollHeight;
                };
            }

            send.onclick = () => {
                const message = input.value.trim();
                if (!message || !ws || ws.readyState !== 1) return;
                renderBubble('user', message);
                streamingBubble = null;
                ws.send(message);
                input.value = '';
            };

            input.addEventListener('input', () => {
                send.disabled = !input.value.trim();
            });

            newChat.onclick = createSession;

            (async () => {
                await fetchSessions();
                if (sessionId) {
                    await loadSession(sessionId);
                } else {
                    await createSession();
                }
            })();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
