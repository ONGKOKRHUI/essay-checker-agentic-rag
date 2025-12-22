# Web search and crawler implementations for agents.
from langchain.tools import tool
from src.database.vector_ops import setup_knowledge_base

# Initialize retriever once
retriever = setup_knowledge_base()

@tool 
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base of relevant essay documents."""
    if not retriever:
        return "Knowledge base is empty."
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])