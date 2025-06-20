import pytest
import asyncio
from app.agents.llm_agent import LLMAgent
from app.memory.rag import RAGMemory

@pytest.mark.asyncio
async def test_llm_agent_strict_doc_qa():
    rag = RAGMemory()
    doc = "The capital of France is Paris."
    rag.add(doc)
    agent = LLMAgent()
    state = {"query": "What is the capital of France?", "rag_memory": rag}
    result = await agent.run(state)
    assert "Paris" in result["result"] 