import pytest
import asyncio
from app.agents.master_agent import MasterAgent

@pytest.mark.asyncio
async def test_master_agent_end_to_end():
    agent = MasterAgent()
    task = "Summarize this: 'LangGraph is a library for LLM agent workflows.' Then calculate 2+2."
    state = {"query": task}
    result = await agent.run(state)
    assert "LangGraph" in str(result)
    assert "4" in str(result) 