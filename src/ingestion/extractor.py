# Fact Extractor 
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import OPENAI_API_KEY, SILICON_FLOW_BASE_URL

# --- Pydantic Schemas ---
class FactsInfo(BaseModel):
    statement: str = Field(
            ..., 
            description="The factual claim made by the student."
        )
    source_quote: str = Field(
            ..., 
            description="The exact sentence or phrase from the essay where this fact is mentioned."
        )
    page_number: int = Field(
            ..., 
            description="The page number where this fact was found."
        )

class FactExtraction(BaseModel):
    facts: List[FactsInfo]

# --- Main Logic ---
def extract_facts_from_docs(docs: list, callbacks=None):
    """
    Extracts facts from essay content and outputs a list of dictionaries.
    """
    # Setup the Model (DeepSeek via OpenAI API) ---
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=SILICON_FLOW_BASE_URL,
        temperature=0  # Keep it 0 for consistent analysis
    )

    # Bind the schema to the model
    structured_llm = llm.with_structured_output(FactExtraction)

    system_prompt = """
    You are an expert fact-checker. 
    Extract every distinct factual claim made in the text provided.
    Ignore opinions or transitional phrases.
    For every fact, you must provide the exact quote from the text.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    extraction_chain = prompt | structured_llm
    
    all_facts_with_metadata = []

    print(f"Processing {len(docs)} pages for fact extraction...")

    for doc in docs:
        page_num = doc.metadata.get('page', 0) + 1
        page_content = doc.page_content
        
        # Skip empty pages to save API calls
        if not page_content.strip():
            continue

        try:
            # We only send the text to the LLM
            result = extraction_chain.invoke({"text": page_content}, config={"callbacks": callbacks})
            print("Extracted facts from page", page_num, "out of", len(docs), "pages")

            # We attach the page number manually here.
            if result and result.facts:
                for fact in result.facts:
                    fact_dict = fact.model_dump() # Updated from .dict()
                    fact_dict['page_number'] = page_num
                    all_facts_with_metadata.append(fact_dict)
            print("Page", page_num, "has", len(result.facts), "facts")

        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            
    return all_facts_with_metadata