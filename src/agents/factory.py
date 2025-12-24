# agent initialization
import json
import asyncio
from typing import Literal, List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.observability import get_langfuse_handler

from src.config import OPENAI_API_KEY, SILICON_FLOW_BASE_URL, JINA_API_KEY
from src.agents.tools import search_knowledge_base

# --- Schemas ---
class FactEvaluation(BaseModel):
    statement: str = Field(description="The verbatim excerpt of the statement")
    correctness_score: Literal["correct", "wrong", "undetermined"] = Field(description="The final verdict")
    summary_description: str = Field(description="Summary of why this verdict was reached")
    #search_results: str = Field(description="A short verbatim of the search results")
    source_document: str = Field(description="One source document - if it is from knowledge base, output the *document name* and *page number*, if it is from web, output the *URL*")

# --- System Prompt ---
SYSTEM_PROMPT = f"""
You are a fact-checking assistant that verifies a single factual statement.

You MUST follow the tool usage and decision rules exactly.

====================
TOOL USAGE RULES
====================

1. You MUST call `search_knowledge_base` exactly ONCE as your first step.
2. After reviewing the retrieved content:
   - If the content clearly SUPPORTS the statement → verdict = "correct"
   - If the content clearly CONTRADICTS the statement → verdict = "wrong"
   - If the content is unrelated, neutral, ambiguous, or missing key information → verdict is NOT decided yet
3. ONLY if the knowledge base is insufficient as defined above, you MAY call `search_web` at most ONCE.
4. After web search:
   - If web content clearly supports → "correct"
   - If web content clearly contradicts → "wrong"
   - If still unclear or conflicting → "undetermined"
5. You MUST NOT call any tool more than once.

====================
DEFINITIONS
====================

- "Supports": Explicitly states the same fact without contradiction.
- "Contradicts": Explicitly states the opposite of the fact.
- "Insufficient": Mentions the topic but does not confirm or deny the exact claim.

====================
OUTPUT FORMAT
====================

Return ONLY a valid JSON object that strictly matches this schema:

{FactEvaluation.model_json_schema()}

====================
SOURCE DOCUMENT RULES
====================

- If the verdict is based on the knowledge base:
  - source_document MUST be: "<document_name>, page <page_number>"
- If the verdict is based on web search:
  - source_document MUST be a single URL
- If verdict is "undetermined":
  - source_document MUST be an empty string ""

Do NOT include markdown, explanations outside JSON, or multiple sources.
"""

# --- Agent Runner ---
async def check_facts(facts_list: List[dict]):
    """
    Async function to run the agent over a list of facts.
    """
    # 2. Setup Client pointing to the REMOTE Jina server
    # Note: We filter for 'search' and 'read' tags to save tokens
    client = MultiServerMCPClient({
        "jina": {
            "transport": "streamable_http",
            "url": "https://mcp.jina.ai/v1?include_tags=search,read",
            "headers": {
                "Authorization": f"Bearer {JINA_API_KEY}"
            }
        }
    })

    try:
        # 1. Retrieve the web search and read tools from the MCP server
        print("Connecting to MCP tools...")
        mcp_tools = await client.get_tools()
        all_tools = [search_knowledge_base] + mcp_tools

        # 2. Initialize your LLM
        llm_model = ChatOpenAI(
            model="deepseek-ai/DeepSeek-V3",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=SILICON_FLOW_BASE_URL, 
            temperature=0.2,
            max_tokens=800,
            max_retries=10,  # <--- CRITICAL FIX: Retries aggressively on 429
            request_timeout=60
        )   

        # 3. Create the agent
        agent = create_agent(
            tools=all_tools,    
            system_prompt=SYSTEM_PROMPT,
            model=llm_model,
            name = "FactCheckerAgent",
        )

        callback = get_langfuse_handler()

        # 5. Define Semaphore for Concurrency Control
        # Allows only 3 facts to be processed at the same time
        semaphore = asyncio.Semaphore(3)

        async def process_single_fact(i: int, fact_data: dict):
            """Helper function to process one fact under semaphore protection."""
            statement = fact_data["statement"]
            
            async with semaphore:
                try:
                    # Invoke agent
                    inputs = {"messages": [HumanMessage(content=f"Evaluate this fact: {statement}")]}
                    
                    response = await agent.ainvoke(
                        inputs, 
                        config={
                            "callbacks": [callback],
                            "metadata": {"langfuse_tags": ["agentic-fact-checker"]}
                        }
                    )
                    
                    # Parse structure
                    structured_llm = llm_model.with_structured_output(FactEvaluation)
                    final_eval = await structured_llm.ainvoke(response["messages"][-1].content)
                    
                    print(f"Validated fact {i+1}: {final_eval.correctness_score}")
                    return final_eval.model_dump()
                    
                except Exception as e:
                    print(f"Error validating fact {i+1}: {e}")
                    # Return a safe fallback so the whole batch doesn't crash
                    return {
                        "statement": statement,
                        "correctness_score": "undetermined",
                        "summary_description": f"Error during validation: {str(e)}",
                        "source_document": ""
                    }

        # 6. Run tasks in parallel (controlled)
        print(f"Starting fact check for {len(facts_list)} facts...")
        
        tasks = [process_single_fact(i, fact) for i, fact in enumerate(facts_list)]
        results = await asyncio.gather(*tasks)

        return results
        
    finally:
        # cleanup if needed, MultiServerMCPClient manages context usually
        pass