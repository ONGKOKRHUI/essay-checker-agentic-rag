#Enbedding and querying logic for vector DB
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import OPENAI_API_KEY, SILICON_FLOW_BASE_URL, VECTOR_DB_PATH, KB_DIR

def setup_knowledge_base():
    """
    Ingests PDFs from the knowledge base directory, splits them, 
    and returns a retriever. Uses persistent storage.
    """
    
    # define embeddings model
    # SiliconFlow hosts open-source embedding models that can be used with LangChain
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=OPENAI_API_KEY,
        base_url=SILICON_FLOW_BASE_URL,
        # Crucial for SiliconFlow/Local providers to avoid dimension errors
        check_embedding_ctx_length=False,
        chunk_size=64
    )

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        print("Loading existing Vector Store...")
        vectorstore = Chroma(
            persist_directory=str(VECTOR_DB_PATH),
            embedding_function=embeddings,
            collection_name="essay_kb")
    else:
        print("Creating new Knowledge Base from documents...")
        # load knowledge base documents
        loader = DirectoryLoader(str(KB_DIR), glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        
        if not docs:
            print("Warning: No documents found in Knowledge Base folder.")
            return None

        # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                collection_name="essay_kb",
                persist_directory=str(VECTOR_DB_PATH)
            )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever