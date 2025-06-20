import pytest
from app.tools.calculator import CalculatorTool

def test_calculator_addition():
    calc = CalculatorTool()
    result = calc.run("32+64")
    assert "96" in result

def test_calculator_complex():
    calc = CalculatorTool()
    result = calc.run("32 + sqrt(223) + abs(-22) / 123")
    # Check for numeric result (roughly)
    assert any(s in result for s in ["47.11", "47.112"])  # Accepts float rounding 