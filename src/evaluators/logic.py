# Logic scoring 
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import OPENAI_API_KEY, SILICON_FLOW_BASE_URL

# --- Schemas ---
class LogicalFallacy(BaseModel):
    fallacy_type: str = Field(..., description="Name of the fallacy.")
    location_snippet: str = Field(..., description="The quote from the text containing the fallacy.")
    explanation: str = Field(..., description="Why this argument is logically flawed.")

class RelevanceAnalysis(BaseModel):
    is_off_topic: bool = Field(..., description="True if the essay completely fails to address the prompt.")
    score: int = Field(..., description="Score 1-10.")
    thesis_alignment: str = Field(..., description="Analysis of whether the thesis statement addresses the prompt.")
    missing_key_points: List[str] = Field(..., description="List of key concepts missing.")

class StructureAnalysis(BaseModel):
    has_clear_intro: bool = Field(..., description="Does it have a distinct introduction?")
    has_clear_conclusion: bool = Field(..., description="Does it have a distinct conclusion?")
    flow_score: int = Field(..., description="Score 1-10.")
    structural_weaknesses: List[str] = Field(..., description="List of specific structural issues.")

class LogicAnalysisResult(BaseModel):
    relevance: RelevanceAnalysis = Field(..., description="Analysis of how well the essay answers the prompt.")
    structure: StructureAnalysis = Field(..., description="Analysis of the essay's organization.")
    identified_fallacies: List[LogicalFallacy] = Field(default_factory=list, description="List of logical errors.")
    argument_strength_score: int = Field(..., description="Score 1-10 on overall persuasiveness.")
    summary_critique: str = Field(..., description="A concise summary of the logical quality.")

# --- Logic ---
def check_logic(essay_text: str, essay_question: str, callbacks=None):
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=SILICON_FLOW_BASE_URL,
        temperature=0
    )
    structured_llm = llm.with_structured_output(LogicAnalysisResult)

    system_prompt = """
    You are a strict Essay Editor and Logic Expert.
    Your task is to ruthlessly evaluate the **Relevance** and **Logic** of the student's essay against the provided Question.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Essay Question: {question}\n\nStudent Essay Content:\n{essay_content}")
    ])

    chain = prompt | structured_llm
    
    print(f"Analyzing Logic...")
    try:
        result = chain.invoke({
            "question": essay_question, 
            "essay_content": essay_text}, 
            config={"callbacks": callbacks})
        return result.model_dump()
    except Exception as e:
        print(f"Error during logic analysis: {e}")
        return None