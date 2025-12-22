# entry point: Orchestrates the pipeline
import asyncio
import json
import logging
from src.config import (
    ESSAY_PDF_PATH, 
    QUESTION_PDF_PATH, 
    RUBRIC_PDF_PATH,
    FACTS_JSON_PATH,
    RUBRICS_JSON_PATH,
    FACT_CHECK_OUTPUT_PATH,
    LOGIC_OUTPUT_PATH,
    LANGUAGE_OUTPUT_PATH,
    FINAL_REPORT_PATH
)
from src.ingestion.pdf_loader import load_pdf, load_pdf_as_text
from src.ingestion.extractor import extract_facts_from_docs
from src.evaluators.rubrics import extract_rubric_data
from src.evaluators.logic import check_logic
from src.evaluators.language import check_language
from src.agents.factory import check_facts
from src.output.llm_judge import generate_final_report

async def main():
    print("🚀 Starting Essay Checker Agentic Pipeline...\n")

    # 1. Load Documents
    essay_docs = load_pdf(ESSAY_PDF_PATH)
    essay_text = load_pdf_as_text(ESSAY_PDF_PATH)
    question_text = load_pdf_as_text(QUESTION_PDF_PATH)
    rubric_text = load_pdf_as_text(RUBRIC_PDF_PATH)

    # 2. Extract Rubric Criteria
    rubric_data = extract_rubric_data(rubric_text)
    with open(RUBRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rubric_data, f, indent=2)

    # 3. Logic & Relevance Check
    logic_data = check_logic(essay_text, question_text)
    with open(LOGIC_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(logic_data, f, indent=2)

    # 4. Language & Grammar Check
    language_data = check_language(essay_text)
    with open(LANGUAGE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(language_data, f, indent=2)

    # 5. Extract Facts & Run Agentic Fact Checker
    raw_facts = extract_facts_from_docs(essay_docs)
    with open(FACTS_JSON_PATH, "w", encoding="utf-8") as f:
        for fact in raw_facts:
            f.write(json.dumps(fact) + "\n")
    
    # Run Async Fact Checker
    verified_facts = await check_facts(raw_facts)
    with open(FACT_CHECK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(verified_facts, f, indent=2)

    # 6. Final Judge (Synthesize Report)
    final_report = generate_final_report(
        essay_content=essay_text,
        essay_question=question_text,
        rubric_data=rubric_data,
        logic_data=logic_data,
        fact_data=verified_facts,
        language_data=language_data
    )

    # 7. Save Report
    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(final_report)
    
    print(f"\n✅ Pipeline Complete! Report saved to: {FINAL_REPORT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())