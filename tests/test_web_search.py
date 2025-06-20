from app.tools.web_search import WebSearch

def test_web_search_format():
    search = WebSearch()
    result = search.search("latest news on AI")
    assert isinstance(result, dict)
    assert "results" in result
    assert isinstance(result["results"], list)
    # Optionally check at least one result
    assert len(result["results"]) > 0 