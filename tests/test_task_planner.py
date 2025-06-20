import pytest
import asyncio
from app.agents.task_planner_agent import TaskPlannerAgent

@pytest.mark.asyncio
async def test_task_planner_plan():
    planner = TaskPlannerAgent()
    task = "Summarize the document and calculate the average."
    state = {"query": task}
    result = await planner.run(state)
    print("TaskPlannerAgent result:", result)
    assert isinstance(result, dict)
    assert "steps" in result
    assert any("summarize" in step.get("tool", "") or step.get("tool", "") == "summarizer" for step in result["steps"])
    assert any("calculator" in step.get("tool", "") for step in result["steps"]) 