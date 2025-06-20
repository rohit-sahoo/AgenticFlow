"""
Web search agent for retrieving information from the internet.
"""
from typing import Any, Dict
import json
from app.agents.base_agent import BaseAgent
from app.tools.web_search import WebSearch

class WebSearchAgent(BaseAgent):
    """Agent for performing web searches."""
    
    def __init__(self):
        """Initialize the web search agent."""
        super().__init__(name="web_search")
        self.search_tool = WebSearch()
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a web search.
        
        Args:
            state: The current state containing:
                - query: The search query
                
        Returns:
            Updated state with search results added
        """
        query = state.get("query", "")
        if not query:
            state["result"] = "Error: No search query provided"
            state["success"] = False
            return state
        
        try:
            result = self.search_tool.search(query)
            formatted_result = self._format_results(result)
            state["result"] = formatted_result
            state["raw_result"] = result
            state["success"] = True
        except Exception as e:
            state["result"] = f"Web search error: {str(e)}"
            state["success"] = False
        
        return state
    
    def _format_results(self, result: Any) -> str:
        """Format web search results for display."""
        try:
            if isinstance(result, dict) and "results" in result:
                data = result
            elif isinstance(result, str):
                try:
                    data = json.loads(result)
                except:
                    return f"**Web Search Results:**\n\n{result}"
            else:
                return f"**Web Search Results:**\n\n{str(result)}"
            
            if not isinstance(data, dict) or "results" not in data:
                return f"**Web Search Results:**\n\n{str(result)}"
            
            output = ["**Web Search Results:**\n"]
            for item in data["results"]:
                if isinstance(item, dict):
                    title = item.get("title", "(No Title)")
                    url = item.get("url", "")
                    content = item.get("content", "")
                    if len(content) > 800:
                        content = content[:800] + "..."
                    if url:
                        output.append(f"- [{title}]({url})\n  {content}")
                    else:
                        output.append(f"- {title}\n  {content}")
            return "\n".join(output)
        except Exception as e:
            return f"**Web Search Results:**\n\n{str(result)}" 