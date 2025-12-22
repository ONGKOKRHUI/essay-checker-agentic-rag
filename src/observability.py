from langfuse.langchain import CallbackHandler
from src.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

def get_langfuse_handler():
    """
    Returns a configured Langfuse CallbackHandler.
    Returns None if keys are missing (to prevent crashing in dev).
    """
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print("⚠️ Langfuse keys not found. Tracing disabled.")
        return None
        
    return CallbackHandler()