"""
Document summarization agent for summarizing text documents.
"""
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.tools.summarizer import DocumentSummarizerTool
from langchain_groq import ChatGroq

class SummarizerAgent(BaseAgent):
    """Agent for summarizing documents."""
    
    def __init__(self):
        """Initialize the summarizer agent."""
        super().__init__(name="summarizer")
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            temperature=0.0,
            max_tokens=4096
        )
        self.summarizer = DocumentSummarizerTool(self.llm)
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize a document.
        
        Args:
            state: The current state containing:
                - document: The document text to summarize
                
        Returns:
            Updated state with document summary added
        """
        document = state.get("document", "")
        if not document:
            state["result"] = "No document is currently available to summarize."
            state["success"] = False
            return state
        
        try:
            result = self.summarizer.run(document)
            state["result"] = f"**Document Summary:**\n{result}"
            state["success"] = True
        except Exception as e:
            state["result"] = f"Summarization error: {str(e)}"
            state["success"] = False
        
        return state 