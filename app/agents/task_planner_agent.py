"""
Task planner agent for breaking down complex tasks into steps.
"""
from typing import Any, Dict, List, Union
import json
import re
from app.agents.base_agent import BaseAgent
from langchain_groq import ChatGroq

class TaskPlannerAgent(BaseAgent):
    """Agent for planning tasks and deciding which other agents to use."""
    
    def __init__(self):
        """Initialize the task planner agent."""
        super().__init__(name="task_planner")
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            temperature=0.0,
            max_tokens=4096,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan a task by breaking it down into steps.
        
        Args:
            state: The current state containing:
                - query: The user's query
                - document: Optional document text
                
        Returns:
            Updated state with task steps added
        """
        query = state.get("query", "")
        document = state.get("document", "")
        
        if not query:
            state["steps"] = []
            state["success"] = False
            state["error"] = "No query provided"
            return state
            
        try:
            # Create a system prompt for the task planner
            system_prompt = """
            You are a professional task planner. Your job is to break down a user's request into a series of 
            sequential steps, each using one of the available tools. Return your answer as a JSON array.
            
            Available tools:
            - "llm": Use for general questions, conversation, and reasoning (including document Q&A)
            - "calculator": Use for mathematical calculations
            - "web_search": Use for searching the internet
            - "code_executor": Use for executing Python code
            - "summarizer": Use ONLY for summarizing a document (when the user says 'summarize' or 'summary')
            
            For each step, specify:
            1. "tool": The name of the tool to use
            2. "input": The specific input for that tool
            
            Example response format:
            {
                "steps": [
                    {"tool": "web_search", "input": "latest news about AI"},
                    {"tool": "calculator", "input": "5 * 10 + 3"},
                    {"tool": "code_executor", "input": "print('Hello world')"},
                    {"tool": "summarizer", "input": "document"},
                    {"tool": "llm", "input": "What are the analytics in the document?"},
                    {"tool": "llm", "input": "Who is the author of the document?"}
                ]
            }
            
            IMPORTANT:
            - For any mathematical calculation, always use a single calculator step with the full expression, e.g. {"tool": "calculator", "input": "42 * sqrt(19) + abs(-7) / 3"}. Do NOT split calculations into multiple steps.
            - For queries like 'add X to the previous result', always use the last calculator result from memory and generate a single calculator step with the correct expression. **ALWAYS use the full-precision previous result, including all decimal places (e.g., if the last result was 185.407089, and the user says 'add 10 to the previous result', generate {"tool": "calculator", "input": "185.407089+10"}). Never use a rounded or truncated value.**
            - Do NOT add unrelated steps (web search, summarizer, code, etc.) for simple math follow-ups.
            - Only use the summarizer tool if the user explicitly asks for a summary (e.g., 'summarize', 'give me a summary', 'summary of the document').
            - **Never add a summarizer step unless the user query contains the word 'summarize' or 'summary'.**
            - For all other questions about the document (e.g., 'what analytics are present?', 'who is the author?', 'list all companies mentioned'), use only the LLM tool (with RAG), and do NOT add a summarizer step.
            - For document Q&A (any question about the document that is NOT a summary), use the 'llm' tool and pass the question as input.
            - **Never add a calculator step for code execution requests, even if the code contains numbers or math. Only use code_executor for code blocks or code-related queries.**
            
            Positive examples:
            - User: "add 10 to the previous result" (previous result was 185.407089) → {"tool": "calculator", "input": "185.407089+10"}
            - User: "what analytics is present in the document?" → {"tool": "llm", "input": "what analytics is present in the document?"}
            - User: "summarize the document" → {"tool": "summarizer", "input": "document"}
            - User: "run this code: print('Hello, world!')" → {"tool": "code_executor", "input": "print('Hello, world!')"}
            - User: "run this code: from functools import lru_cache\n..." → {"tool": "code_executor", "input": "from functools import lru_cache\n..."}
            
            Negative examples:
            - User: "add 10 to the previous result" (previous result was 185.407089) → {"tool": "calculator", "input": "185+10"} (INCORRECT: do not round or truncate)
            - User: "what analytics is present in the document?" → {"tool": "summarizer", "input": "document"} (INCORRECT: do not add a summarizer step for Q&A)
            - User: "run this code: print('Hello, world!')" → [{"tool": "code_executor", "input": "print('Hello, world!')"}, {"tool": "calculator", "input": "print('Hello, world!')"}] (INCORRECT: do not add a calculator step for code execution)
            - User: "run this code: from functools import lru_cache\n..." → [{"tool": "code_executor", "input": "from functools import lru_cache\n..."}, {"tool": "calculator", "input": "from functools import lru_cache\n..."}] (INCORRECT: do not add a calculator step for code execution)
            
            If the query is a simple greeting or a single question, you may return just one step.
            For code execution, extract ONLY the code, removing any markdown formatting.
            """
            
            # Add document context if available
            if document:
                system_prompt += f"\n\nA document has been provided. For document-related tasks, use the summarizer tool."
            
            # Create the full prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the JSON response
            try:
                parsed = json.loads(str(content))
                if isinstance(parsed, dict) and "steps" in parsed:
                    steps = parsed["steps"]
                elif isinstance(parsed, list):
                    steps = parsed
                else:
                    steps = [{"tool": "llm", "input": query}]
            except json.JSONDecodeError:
                # Fallback: Try to extract using regex
                content_str = str(content)
                match = re.search(r'\[\s*\{.*\}\s*\]', content_str, re.DOTALL)
                if match:
                    try:
                        steps = json.loads(match.group(0))
                    except:
                        steps = [{"tool": "llm", "input": query}]
                else:
                    # Final fallback
                    steps = [{"tool": "llm", "input": query}]
            
            # Process code blocks in inputs
            for step in steps:
                if step.get("tool") == "code_executor":
                    input_val = step.get("input", "")
                    # Extract code from markdown if present
                    code = self._extract_code(str(input_val))
                    step["input"] = code
            
            # Post-process: mark LLM steps that are document Q&A (contain 'document' or similar) with a 'rag_qa' flag
            for step in steps:
                if step.get('tool') == 'llm' and any(word in step.get('input', '').lower() for word in ['document', 'analytics', 'content', 'file', 'report', 'paper', 'summary', 'describe', 'explain']):
                    step['rag_qa'] = True
            
            # Return the steps
            # Merge multiple calculator steps into one if needed
            calc_steps = [s for s in steps if s.get('tool') == 'calculator']
            if len(calc_steps) > 1:
                # Try to merge all calculator inputs into a single expression
                merged_expr = ' ; '.join(s['input'] for s in calc_steps)
                steps = [s for s in steps if s.get('tool') != 'calculator']
                steps.append({'tool': 'calculator', 'input': merged_expr})
            state["steps"] = steps
            state["success"] = True
            return state
            
        except Exception as e:
            # Fallback to single LLM step
            state["steps"] = [{"tool": "llm", "input": query}]
            state["success"] = False
            state["error"] = f"Task planning error: {str(e)}"
            return state
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks."""
        # Try to find markdown code blocks
        patterns = [
            r'```(?:python)?\s*\n(.*?)```',  # ```python\ncode```
            r'```(?:python)?\s*(.*?)```',     # ```python code```
            r'```(.*?)```'                    # ```code```
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return "\n\n".join(match.strip() for match in matches)
        
        # If no markdown, return the original text
        return text 

if __name__ == "__main__":
    import asyncio
    planner = TaskPlannerAgent()
    for test_query in [
        'Calculate: 42 * sqrt(19) + abs(-7) / 3',
        'add 10 to the previous result',
        'what analytics is present in the document?',
        'summarize the document',
        'who is the author of the document?'
    ]:
        print(f'\n[TEST QUERY] {test_query}')
        state = {'query': test_query}
        result = asyncio.run(planner.run(state))
        print('[DEBUG MAIN] Steps:', result['steps']) 