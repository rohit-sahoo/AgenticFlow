"""
LLM agent for general question answering and conversation.
"""
from typing import Any, Dict, List, Optional
from app.agents.base_agent import BaseAgent
from langchain_groq import ChatGroq
import re

class LLMAgent(BaseAgent):
    """Agent for general question answering and conversation, with RAG Q&A support."""
    
    def __init__(self):
        """Initialize the LLM agent."""
        super().__init__(name="llm")
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            temperature=0.2,
            max_tokens=4096
        )
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query using the LLM, with RAG Q&A if document question.
        
        Args:
            state: The current state containing:
                - query: The user's query
                - chat_history: Optional chat history
                - context: Optional additional context
                - rag_memory: Optional RAG memory instance
                
        Returns:
            Updated state with LLM response
        """
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        context = state.get("context", "")
        rag_memory = state.get("rag_memory", None)
        
        if not query:
            state["result"] = "Error: No query provided"
            state["success"] = False
            return state
        
        # Detect if this is a document question
        if rag_memory and self._is_document_question(query):
            # Retrieve relevant chunks
            rag_results = rag_memory.retrieve(query, k=5)
            rag_content = "\n".join(rag_results)
            if rag_content.strip():
                prompt = (
                    "You are a document Q&A assistant. Use ONLY the provided context to answer the question. "
                    "If the answer is not present, say 'Not found in the document.'\n\n"
                    f"Context:\n---\n{rag_content}\n---\n\nQuestion: {query}\n\nAnswer:"
                )
                messages = [{"role": "user", "content": prompt}]
                response = self.llm.invoke(messages)
                result = response.content if hasattr(response, 'content') else str(response)
                state["result"] = result
                state["success"] = True
                return state
            else:
                state["result"] = "Not found in the document."
                state["success"] = True
                return state
        
        try:
            # Create messages for the LLM
            messages = []
            
            # Add context if available
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"Use the following context to answer the user's query: {context}"
                })
            
            # Add chat history
            for msg in chat_history:
                if isinstance(msg, tuple) and len(msg) == 2:
                    user_msg, ai_msg = msg
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": ai_msg})
            
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            # If no messages were added (which shouldn't happen), add the query
            if not messages:
                messages = [{"role": "user", "content": query}]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content from the response
            result = response.content if hasattr(response, 'content') else str(response)
            
            state["result"] = result
            state["success"] = True
        except Exception as e:
            state["result"] = f"LLM error: {str(e)}"
            state["success"] = False
        
        return state

    def _is_document_question(self, input_val: str) -> bool:
        document_keywords = [
            "document", "metrics", "summary", "content", "file", "upload", "resume", "paper", "report", "company", "work", "experience", "education", "where did", "which company", "who worked", "who is mentioned", "who appears", "who are", "list all", "name all"
        ]
        input_lower = input_val.lower()
        return any(word in input_lower for word in document_keywords) 