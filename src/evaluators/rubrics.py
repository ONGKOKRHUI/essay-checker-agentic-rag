# Rubric extraction logic
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import OPENAI_API_KEY, SILICON_FLOW_BASE_URL

# --- Schemas ---
class PerformanceLevel(BaseModel):
    grade_label: str = Field(..., description="The label (e.g., 'High Distinction', 'A', 'Band 5').")
    score_range: str = Field(..., description="The point range (e.g., '80-100', '16-20').")
    descriptor_points: List[str] = Field(
        ..., 
        description="A list of specific qualifiers/bullets found in this cell."
    )

class AssessmentCriterion(BaseModel):
    category: Optional[str] = Field(
        None, 
        description="The broader category this criterion belongs to (e.g., 'Language')."
    )
    name: str = Field(..., description="The specific skill being assessed.")
    weight: str = Field(..., description="The weight of this criterion (e.g., '30%', '10 marks').")
    levels: List[PerformanceLevel] = Field(..., description="The grading scale for this specific criterion.")

class RubricExtractionResult(BaseModel):
    title: str = Field(..., description="Title of the rubric.")
    context_notes: List[str] = Field(
        default_factory=list, 
        description="Any global rules found."
    )
    criteria: List[AssessmentCriterion]

# --- Logic ---
def extract_rubric_data(rubric_text: str, callbacks=None):
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=SILICON_FLOW_BASE_URL,
        temperature=0
    )
    structured_llm = llm.with_structured_output(RubricExtractionResult)

    system_prompt = """
    You are an expert in Academic Assessment and Pedagogy.
    Your task is to digitize a complex "English for Specific Academic Contexts" (ESAC) rubric.
    
    **Critical Instruction for ESAC Rubrics:**
    1. **Explode Descriptors:** These rubrics often contain dense blocks of text. Separate these into the `descriptor_points` list.
    2. **Capture Categories:** If the rubric groups rows, capture `category`.
    3. **Global Rules:** Look for footnotes or headers about penalties.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    chain = prompt | structured_llm
    
    print("Digitizing Rubric...")
    try:
        result = chain.invoke({"text": rubric_text},
                              config={"callbacks": [callbacks]})
        return result.model_dump()
    except Exception as e:
        print(f"Error during rubric extraction: {e}")
        return None