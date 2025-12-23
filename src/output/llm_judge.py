# Final JSON aggregation and report generation
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.observability import get_langfuse_handler

from src.config import OPENAI_API_KEY, SILICON_FLOW_BASE_URL

def generate_final_report(
    essay_content: str,
    essay_question: str,
    rubric_data: dict,
    logic_data: dict,
    fact_data: list,
    language_data: dict,
    #callbacks=None
    ):

    callback = get_langfuse_handler()

    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=SILICON_FLOW_BASE_URL,
        temperature=0.2 
    )

    system_prompt = """
    You are the **Lead Academic Examiner** for an advanced university course.
    Your task is to grade a student essay by strictly synthesizing data from three expert AI sub-agents (Logic, Fact, Language) and applying a specific Grading Rubric.

    ### 1. INPUT DATA OVERVIEW
    You will receive:
    1.  **Rubric:** The exact criteria and band descriptors.
    2.  **Logic Report:** Scores on relevance, structure, and argument strength.
    3.  **Fact Report:** Verification of claims and citations.
    4.  **Language Report:** Analysis of grammar, vocabulary, and tone.
    5.  **Student Essay:** The raw text.

    ### 2. GRADING ALGORITHM (Mental Steps)
    Before writing the report, perform this analysis:
    * **Step 1 (Relevance Check):** Look at `logic_analysis['relevance']['is_off_topic']`. If TRUE, the maximum score for "Task Response" is capped at **Band 5**.
    * **Step 2 (Map Evidence to Rubric):**
        * *Task Response:* Use `logic_analysis['relevance']` and `logic_analysis['argument_strength_score']`.
        * *Cohesion/Structure:* Use `logic_analysis['structure']['flow_score']` and `language_analysis['structure']['flow_issues']`.
        * *Language/Style:* Use `language_analysis['grammar_issues']` count and `language_analysis['vocabulary']['score']`.
        * *Evidence/Referencing:* Use `fact_checking_output` (look for incorrect citations) and `logic_analysis['identified_fallacies']`.
    * **Step 3 (Select Band):** For each criterion, find the Rubric Level where the `descriptor_points` best match your analysis.

    ### 3. OUTPUT RULES
    * **Tone:** Professional, constructive, and authoritative.
    * **Justification:** In the Scorecard, you MUST quote specific **Descriptor Points** from the rubric to justify the score.
    * **Annotations:** In the "Annotated Text Review", you must strictly use the JSON data to pinpoint errors. Do not hallucinate new errors.

    ### 4. FINAL OUTPUT FORMAT
    Produce a clean Markdown report strictly following the template below.
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        **ESSAY QUESTION:** {essay_question}

        **RUBRIC:** {rubric_json}

        **LOGIC & RELEVANCE REPORT:** {logic_json}

        **FACT CHECK REPORT:** {fact_json}

        **LANGUAGE REPORT:** {language_json}

        **STUDENT ESSAY:** {essay_content}

        ---
        Generate the **Academic Assessment Report** now.
        """)
    ])

    grading_chain = prompt_template | llm | StrOutputParser()

    try:
        report = grading_chain.invoke({
            "essay_question": essay_question,
            "essay_content": essay_content,
            "rubric_json": json.dumps(rubric_data, indent=2),
            "logic_json": json.dumps(logic_data, indent=2),
            "fact_json": json.dumps(fact_data, indent=2),
            "language_json": json.dumps(language_data, indent=2)
        }, config={"callbacks": [callback],
                   "metadata": {"langfuse_tags": ["llm_judge"]},
                   })
        return report
    except Exception as e:
        return f"Error generating report: {e}"