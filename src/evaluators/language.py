# Language and grammar scoring
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.observability import get_langfuse_handler
from src.config import OPENAI_API_KEY, SILICON_FLOW_BASE_URL

# --- Schemas ---
class GrammarError(BaseModel):
    original_text: str = Field(..., description="The exact text snippet containing the error.")
    correction: str = Field(..., description="The corrected version of the text.")
    error_type: str = Field(..., description="Type of error.")
    explanation: str = Field(..., description="Brief explanation.")

class VocabularyAnalysis(BaseModel):
    score: int = Field(..., description="A score from 1-10 rating vocabulary sophistication.")
    repetitive_words: List[str] = Field(..., description="List of words used excessively.")
    advanced_words_used: List[str] = Field(..., description="List of sophisticated words used.")
    feedback: str = Field(..., description="Qualitative feedback.")

class StructureAnalysis(BaseModel):
    sentence_variety_score: int = Field(..., description="A score from 1-10 on sentence variety.")
    flow_issues: List[str] = Field(..., description="List of specific issues with flow.")
    feedback: str = Field(..., description="Qualitative feedback.")

class LanguageAnalysisResult(BaseModel):
    grammar_issues: List[GrammarError] = Field(default_factory=list, description="List of grammar errors.")
    vocabulary: VocabularyAnalysis = Field(..., description="Analysis of vocabulary.")
    structure: StructureAnalysis = Field(..., description="Analysis of sentence structure/flow.")
    overall_tone: str = Field(..., description="Description of the essay's tone.")
    summary_critique: str = Field(..., description="A concise summary of linguistic quality.")

# --- Logic ---
def check_language(essay_text: str):

    callback = get_langfuse_handler()

    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=SILICON_FLOW_BASE_URL,
        temperature=0
    )
    structured_llm = llm.with_structured_output(LanguageAnalysisResult)

    system_prompt = """
    You are a strict Linguistic Professor and Editor. 
    Your goal is to analyze the student's essay purely on **language mechanics, style, and structure**. 
    Do NOT grade the content or arguments; focus only on HOW it is written.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here is the student's essay:\n\n{text}")
    ])

    chain = prompt | structured_llm
    
    print("Analyzing Language...")
    try:
        result = chain.invoke({"text": essay_text}, 
                              config={
                                  "callbacks": [callback], 
                                  "metadata": {"langfuse_tags": ["language-analysis"]},
                                  })
        return result.model_dump()
    except Exception as e:
        print(f"Error during language analysis: {e}")
        return None