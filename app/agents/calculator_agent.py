"""
Calculator agent for evaluating mathematical expressions.
"""
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.tools.calculator import CalculatorTool

class CalculatorAgent(BaseAgent):
    """Agent for evaluating mathematical expressions."""
    
    def __init__(self, memory=None):
        """Initialize the calculator agent."""
        super().__init__(name="calculator")
        self.calculator = CalculatorTool(memory)
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression.
        
        Args:
            state: The current state containing:
                - expression: The mathematical expression to evaluate
                
        Returns:
            Updated state with result added
        """
        expression = state.get("expression", "")
        if not expression:
            state["result"] = "Error: No expression provided"
            state["success"] = False
            return state
        
        try:
            result = self.calculator.run(expression)
            state["result"] = result
            state["success"] = True
        except Exception as e:
            state["result"] = f"Calculator error: {str(e)}"
            state["success"] = False
        
        return state 