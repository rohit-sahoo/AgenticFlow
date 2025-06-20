import os
from langchain_tavily import TavilySearch
from typing import Any

class WebSearch:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        self.search_tool = TavilySearch(tavily_api_key=api_key)

    def search(self, query: str) -> Any:
        result = self.search_tool.run(query)
        return result 