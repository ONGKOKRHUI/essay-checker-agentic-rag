# agent initialization
import json
from typing import Literal, List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config import OPENAI_API_KEY, SILICON_FLOW_BASE_URL, JINA_API_KEY
from src.agents.tools import search_knowledge_base

# --- Schemas ---
class FactEvaluation(BaseModel):
    statement: str = Field(description="The original fact statement")
    correctness_score: Literal["correct", "wrong", "undetermined"] = Field(description="The final verdict")
    summary_description: str = Field(description="Summary of why this verdict was reached")

# --- System Prompt ---
SYSTEM_PROMPT = """You are a fact-checking assistant. 
Logic Flow:
1. For every fact, first use 'search_knowledge_base'.
2. If the retrieved info supports the fact, mark as 'correct'.
3. If it contradicts, mark as 'wrong'.
4. If the knowledge base is insufficient (neutral/unknown), you MUST call 'web_search' ONCE to check online.
5. If web results still don't clarify, mark as 'undetermined'.

Return the result strictly as a JSON object matching the FactEvaluation schema."""

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
        )   

        # 3. Create the agent
        agent = create_agent(
            tools=all_tools,    
            system_prompt=SYSTEM_PROMPT,
            model=llm_model,
            name = "FactCheckerAgent",
        )

        results = []
        print(f"Starting fact check for {len(facts_list)} facts...")
        
        # NOTE: Limited to 5 facts for demonstration speed, remove slice [0:5] for full run
        for i, fact_data in enumerate(facts_list): 
            statement = fact_data["statement"]
            
            # Invoke agent for each fact
            inputs = {"messages": [HumanMessage(content=f"Evaluate this fact: {statement}")]}
            
            # Use structured output parsing
            response = await agent.ainvoke(inputs)
            
            # The last message contains the result. 
            # We can force the agent to return the structured schema
            structured_llm = llm_model.with_structured_output(FactEvaluation)
            final_eval = await structured_llm.ainvoke(response["messages"][-1].content)
            
            results.append(final_eval.model_dump())
            print(f"Validated fact {i+1}: {final_eval.correctness_score}")

        return results
        
    finally:
        # cleanup if needed, MultiServerMCPClient manages context usually
        pass