import os
import httpx
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("JinaSearch")

# Get your key from https://jina.ai/reader/
JINA_API_KEY = os.getenv("JINA_API_KEY")

@mcp.tool()
async def search_web(query: str) -> str:
    """
    Search the web using Jina AI and return the content in clean Markdown.
    Use this for real-time information or deep research.
    """
    if not JINA_API_KEY:
        return "Error: JINA_API_KEY is missing."

    url = f"https://s.jina.ai/{query}"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        return response.text

if __name__ == "__main__":
    mcp.run(transport="stdio")