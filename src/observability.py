from langfuse.langchain import CallbackHandler
from src.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL

def get_langfuse_handler():
    """
    Returns a configured Langfuse CallbackHandler.
    Returns None if keys are missing (to prevent crashing in dev).
    """
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print("⚠️ Langfuse keys not found. Tracing disabled.")
        return None
        
    return CallbackHandler(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_BASE_URL
    )