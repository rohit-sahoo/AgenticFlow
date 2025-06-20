from app.memory.rag import RAGMemory

def test_rag_memory_add_and_query():
    rag = RAGMemory()
    doc = "LangGraph is a library for building LLM agent workflows."
    rag.add(doc)
    result = rag.retrieve("What is LangGraph?")
    assert any("LangGraph" in r for r in result)
    assert any("library" in r for r in result) 