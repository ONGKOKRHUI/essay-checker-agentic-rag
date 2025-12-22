# PDF Processing logic
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str) -> list:
    """
    Loads a PDF and returns a list of documents.
    """
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    if not docs:
        print("Error: No documents found.")
        return []
        
    return docs

def load_pdf_as_text(pdf_path: str) -> str:
    """
    Loads a PDF and merges pages into a single string.
    """
    docs = load_pdf(pdf_path)
    return "\n\n".join([d.page_content for d in docs])