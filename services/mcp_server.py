# mcp_server.py
from fastmcp import FastMCP
import httpx
import asyncio
from services.crawler import process_folder

mcp = FastMCP("QdrantToolbox")
BASE_URL = "http://127.0.0.1:9000" # Your existing FastAPI port


@mcp.tool()
async def add_documents(folder_path: str):
    """
    Crawl all documents in the folder and add them to Qdrant via your crawler.
    """
    try:
        # Run blocking crawler code asynchronously
        result = await asyncio.to_thread(process_folder, folder_path)
        return {"status": "success", "folder": folder_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}



@mcp.tool()
async def search_documents(query: str):
    """Search for relevant documents in Qdrant based on query."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/search",
            json={
                "text": query,     # ✅ CORRECT FIELD
                "k": 5,
                "threshold": 0.1
            }
        )
        return response.json()



@mcp.tool()
async def list_sources(source: str = None):
    """List all indexed files and their hashes for a specific source."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/list", params={"source": source})
        return response.json()

@mcp.tool()
async def init_vector_db():
    """Wipe and re-initialize the Qdrant collection (Warning: Destructive)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/init_qdrant")
        return response.json()

if __name__ == "__main__":
    mcp.run(transport="stdio")
