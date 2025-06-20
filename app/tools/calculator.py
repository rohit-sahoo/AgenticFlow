import math
import re
from typing import Optional

class CalculatorTool:
    """A simple calculator tool for evaluating math expressions safely."""
    def __init__(self, memory=None):
        self.memory = memory
    
    def run(self, expression: str) -> str:
        # Check if this is a context-aware calculation
        if any(word in expression.lower() for word in ["previous", "last", "result", "answer"]):
            if self.memory:
                recent_memory = self.memory.get_recent(5)
                if recent_memory:
                    # Look for recent calculation results
                    lines = recent_memory.split('\n')
                    for line in reversed(lines):
                        if 'AI:' in line and ('Calculator Result:' in line or 'Final Result:' in line or '=' in line):
                            result_match = re.search(r'[-]?\d+\.?\d*', line)
                            if result_match:
                                previous_result = result_match.group(0)
                                new_expression = expression.lower()
                                new_expression = re.sub(r'previous\s+result', previous_result, new_expression)
                                new_expression = re.sub(r'last\s+result', previous_result, new_expression)
                                new_expression = re.sub(r'result', previous_result, new_expression)
                                new_expression = re.sub(r'answer', previous_result, new_expression)
                                return self._evaluate_expression(new_expression)
        return self._evaluate_expression(expression)
    
    def _evaluate_expression(self, expression: str) -> str:
        expression = expression.strip()
        expression = re.sub(r'\s+', '', expression)
        expression = re.sub(r'add', '+', expression, flags=re.IGNORECASE)
        expression = re.sub(r'subtract', '-', expression, flags=re.IGNORECASE)
        expression = re.sub(r'multiply', '*', expression, flags=re.IGNORECASE)
        expression = re.sub(r'divide', '/', expression, flags=re.IGNORECASE)
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names['abs'] = abs
        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            # Format result: up to 6 decimals, trim trailing zeros and dot
            if isinstance(result, float):
                formatted = f"{result:.6f}".rstrip('0').rstrip('.')
            else:
                formatted = str(result)
            return f"{expression} = {formatted}"
        except Exception as e:
            return f"Calculator error: {e}" 