"""
Code execution agent for running Python code.
"""
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.tools.code_executor import CodeExecutor
from app.agents.llm_agent import LLMAgent
import re

class CodeAgent(BaseAgent):
    """Agent for executing Python code with persistent namespace and auto-fix."""
    
    def __init__(self):
        """Initialize the code execution agent."""
        super().__init__(name="code_executor")
        self.executor = CodeExecutor()
        self.llm_agent = LLMAgent()
        # Persistent namespace for the session
        self.session_namespace = {}
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Python code with persistent namespace and auto-fix.
        
        Args:
            state: The current state containing:
                - code: The Python code to execute
                
        Returns:
            Updated state with execution result added
        """
        code = state.get("code", "")
        if not code:
            state["result"] = "Error: No code provided"
            state["success"] = False
            return state
        
        # Clean and extract code block
        code = self._extract_code_from_message(code)
        
        # Try to execute the code in the persistent namespace
        result = self._run_code_with_namespace(code)
        
        # If syntax or indentation error, try to auto-fix with LLM
        if any(err in result for err in ["SyntaxError", "IndentationError"]):
            fixed_code = await self._fix_code_with_llm(code)
            if fixed_code.strip() != code.strip():
                result = self._run_code_with_namespace(fixed_code)
                if "Error" not in result:
                    state["result"] = f"🤖 Auto-fixed syntax. **Result:**\n{result}"
                    state["success"] = True
                    return state
            state["result"] = f"**Code Execution Error:** {result} (Auto-fixer failed)"
            state["success"] = False
            return state
        
        if "Error" in result:
            state["result"] = f"**Code Execution Error:** {result}"
            state["success"] = False
            return state
        
        state["result"] = f"**Code Execution Result:**\n{result}"
        state["success"] = True
        return state

    def _run_code_with_namespace(self, code: str) -> str:
        import io
        import contextlib
        output = io.StringIO()
        try:
            code = code.strip()
            if not code:
                return "(No code provided)"
            # Use persistent namespace for the session
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(code, self.session_namespace, self.session_namespace)
            result = output.getvalue().strip()
            # If no output, check for variables
            if not result:
                for name, value in self.session_namespace.items():
                    if not name.startswith('_') and not callable(value):
                        if isinstance(value, (int, float, str, list, dict, tuple, set)):
                            result = f"Variable '{name}' = {value}"
                            break
            return result or "(Code executed successfully, no output produced)"
        except RecursionError as e:
            return f"Recursion error: {e}. The function may have infinite recursion."
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        except Exception as e:
            return f"Error: {e.__class__.__name__}: {e}"

    def _extract_code_from_message(self, message: str) -> str:
        # Extract code from markdown or plain text
        markdown_patterns = [
            r'```(?:python)?\s*\n(.*?)```',
            r'```(?:python)?\s*(.*?)```',
            r'```(.*?)```',
        ]
        for pattern in markdown_patterns:
            matches = re.findall(pattern, message, re.DOTALL)
            if matches:
                return "\n\n".join(match.strip() for match in matches)
        # If no markdown, return as is
        return message.strip()

    async def _fix_code_with_llm(self, code: str) -> str:
        # Use the LLM agent to auto-fix code
        prompt = (
            "You are a Python code fixer. The following Python code has syntax or indentation errors. "
            "Fix all issues and return ONLY the corrected code without any explanation, markdown formatting, or additional text.\n\n"
            f"Broken Code:\n{code}\n\nFixed Code (return only the code):"
        )
        llm_state = {"query": prompt}
        response = await self.llm_agent.run(llm_state)
        fixed_code = response.get("result", "").strip()
        # Remove markdown if present
        if fixed_code.startswith("```python"):
            fixed_code = fixed_code[9:]
        elif fixed_code.startswith("```"):
            fixed_code = fixed_code[3:]
        if fixed_code.endswith("```"):
            fixed_code = fixed_code[:-3]
        return fixed_code.strip() 