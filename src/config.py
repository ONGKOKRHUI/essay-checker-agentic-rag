#API keys, DB urls, Model parameters

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SILICON_FLOW_BASE_URL = os.getenv("SILICON_FLOW_BASE_URL", "https://api.siliconflow.cn/v1")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Input/Raw Paths
RAW_DIR = DATA_DIR / "raw"
ESSAY_PDF_PATH = RAW_DIR / "essay_content.pdf"
QUESTION_PDF_PATH = RAW_DIR / "essay_question.pdf"
RUBRIC_PDF_PATH = RAW_DIR / "essay_rubric.pdf"

# Knowledge Base Paths
KB_DIR = DATA_DIR / "knowledge_base"
VECTOR_DB_PATH = DATA_DIR / "chroma_db"

# Output/Processed Paths
PROCESSED_DIR = DATA_DIR / "processed"
FACTS_JSON_PATH = PROCESSED_DIR / "extracted_facts.jsonl"
RUBRICS_JSON_PATH = PROCESSED_DIR / "extracted_rubrics.json"
FACT_CHECK_OUTPUT_PATH = PROCESSED_DIR / "fact_checking_output.json"
LOGIC_OUTPUT_PATH = PROCESSED_DIR / "logic_analysis_output.json"
LANGUAGE_OUTPUT_PATH = PROCESSED_DIR / "language_analysis_output.json"
FINAL_REPORT_PATH = DATA_DIR / "final_report/final_report.md"

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "final_report").mkdir(parents=True, exist_ok=True)

# Langfuse Config
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")