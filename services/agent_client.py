import sys
import asyncio
import os
import json
import hashlib
from typing import Annotated, TypedDict, Optional
from dotenv import load_dotenv

# 1. WINDOWS/CONDA FIX
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

# --- State Definition ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

async def run_agent():
    # 2. Setup MCP Client
    # Start MCP server as a module
    client = MultiServerMCPClient({
        "qdrant_tools": {
            "command": sys.executable,
            "args": ["-m", "services.mcp_server"],
            "transport": "stdio"
        }
    })
    
    print("🤖 Initializing tools from MCP Server...")
    tools = await client.get_tools()
    
    # 3. REFINED SYSTEM PROMPT (Optimized for Groq Tool Calling)
    # We remove the aggressive 'STRICT' language which often causes the 400 error.
    system_prompt = SystemMessage(content=(
        "You are a helpful Research Assistant with access to a document database. "
        "For general knowledge questions not related to the documents, answer directly without using any tools. "
        "Only for questions specifically about documents or data in the database, use the 'search_documents' tool. "
        "Provide a natural summary of the findings. "
        "Always include the 'Source Name' and 'Relevance Score' from the metadata in the format: "
        "Source Name: [source] "
        "Document Title: [doc_title] "
        "Filename: [filename] "
        "Relevance Scores: [range, e.g., 0.2 to 0.4] "
        "If the search tool returns no results, tell the user you couldn't find information in the documents."
    ))

    # 4. Initialize Groq
    # Using Llama 3.1 8B for better tool calling reliability
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0, 
        groq_api_key=os.getenv("GROQ_API_KEY")
    ).bind_tools(tools)

    # 5. Build the Graph
    workflow = StateGraph(State)

    def call_model(state: State):
        # We combine the system prompt with the message history
        response = model.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    workflow.add_node("agent", call_model)
    workflow.add_node("action", ToolNode(tools))

    workflow.add_edge(START, "agent")
    
    def should_continue(state: State):
        last_message = state["messages"][-1]
        # Check if the model wants to call a tool
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "action"
        return END

    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("action", "agent")

    app = workflow.compile()

    print("\n✅ Agent Ready! Ask about your documents.")
    chat_history = []
    
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\n👤 You: ")
            if user_input.lower() in ["exit", "quit", "q"]: break

            chat_history.append(HumanMessage(content=user_input))
            
            final_text = ""
            print("⏳ Thinking...", end="\r")
            
            # 6. Execute Graph with Tool Monitoring
            async for event in app.astream({"messages": chat_history}, stream_mode="values"):
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    
                    # Explicitly catch and show tool usage
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            print(f"\n🛠️  [TOOL CALL]: {tool_call['name']}(query='{tool_call['args'].get('query')}')")
                    
                    # Final response from the agent node after tool output is processed
                    if isinstance(last_msg, AIMessage) and last_msg.content and not last_msg.tool_calls:
                        final_text = last_msg.content

            if final_text:
                print(f"🤖 Assistant: {final_text}")
                chat_history.append(AIMessage(content=final_text))

        except Exception as e:
            print(f"\n❌ Execution Error: {str(e)}")
            print("💡 Tip: Try restarting your FastAPI backend and MCP server.")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: Set GROQ_API_KEY in your .env file")
    else:
        asyncio.run(run_agent())