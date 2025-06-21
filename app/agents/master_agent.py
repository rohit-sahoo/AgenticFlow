"""
Master agent for orchestrating the workflow using LangGraph.
"""
from typing import Any, Dict, List, Tuple, cast, Annotated, TypedDict, Optional, Union
import os
import asyncio
from langgraph.graph import StateGraph, END
from datetime import datetime

from app.agents.base_agent import BaseAgent
from app.agents.task_planner_agent import TaskPlannerAgent
from app.agents.calculator_agent import CalculatorAgent
from app.agents.code_agent import CodeAgent
from app.agents.web_search_agent import WebSearchAgent
from app.agents.summarizer_agent import SummarizerAgent
from app.agents.llm_agent import LLMAgent
from app.memory import ShortTermMemory, RAGMemory
from app.agents.flow_logger import FlowLogger

# Define the state type for typechecking
class StepInfo(TypedDict):
    tool: str
    input: str
    status: str
    result: Optional[str]

class AgentState(TypedDict):
    query: str
    steps: List[StepInfo]
    current_step_index: int
    results: List[str]
    failed_steps: List[Dict[str, Any]]
    document: str
    chat_history: List[Tuple[str, str]]
    final_response: Optional[str]

class MasterAgent(BaseAgent):
    """Master agent that orchestrates the workflow using LangGraph."""
    
    def __init__(self, memory=None, rag_memory=None, long_memory=None):
        """Initialize the master agent."""
        super().__init__(name="master")
        self.memory = memory or ShortTermMemory()
        self.rag_memory = rag_memory or RAGMemory()
        self.long_memory = long_memory
        
        # Initialize sub-agents
        self.task_planner = TaskPlannerAgent()
        self.calculator = CalculatorAgent(memory)
        self.code_executor = CodeAgent()
        self.web_search = WebSearchAgent()
        self.summarizer = SummarizerAgent()
        self.llm = LLMAgent()
        
        # Build the LangGraph
        self.graph = self._build_graph()
        self.logger = FlowLogger()
        self.logger.log_event("MasterAgent initialized and flow started.")
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent workflow.
        
        Args:
            state: The initial state
                
        Returns:
            Final state with results
        """
        user_input = state.get('query', '')
        self.logger.log_user_input()
        if user_input:
            self.logger.log(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - [USER INPUT] - {user_input}")
        self.logger.log_event(f"MasterAgent triggered with query: {state.get('query', '')}")
        result = await self.graph.ainvoke({
            "query": state.get("query", ""),
            "document": state.get("document", ""),
            "chat_history": state.get("chat_history", []),
            "steps": [],
            "current_step_index": 0,
            "results": [],
            "failed_steps": [],
            "final_response": None,
            "logger": self.logger
        })
        
        # Update the state with the results
        state.update(result)
        
        # Make sure final_response exists
        if "final_response" not in state or not state["final_response"]:
            state["final_response"] = "No response generated."
        
        # Update memory with conversation history
        if self.memory and "query" in state and "final_response" in state:
            self.memory.add(state["query"], state["final_response"])
        
        self.logger.log_event("MasterAgent finished.")
        final_response = state.get('final_response', '')
        if final_response:
            self.logger.log_final_response(final_response)
        return state
    
    def _build_graph(self) -> Any:
        """Build the LangGraph for orchestrating the agents."""
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes for each step in the workflow - ensure we're using synchronous functions
        graph.add_node("task_planner", self._task_planner_node_sync)
        graph.add_node("execute_calculator", self._calculator_node_sync)
        graph.add_node("execute_code", self._code_executor_node_sync)
        graph.add_node("execute_web_search", self._web_search_node_sync)
        graph.add_node("execute_summarizer", self._summarizer_node_sync)
        graph.add_node("execute_llm", self._llm_node_sync)
        graph.add_node("next_step", self._next_step_node_sync)
        graph.add_node("format_response", self._format_response_node_sync)
        
        # Set the entry point to task planning
        graph.set_entry_point("task_planner")
        
        # Define edges for task execution
        graph.add_conditional_edges(
            "task_planner",
            self._route_to_tool,
            {
                "calculator": "execute_calculator",
                "code_executor": "execute_code",
                "web_search": "execute_web_search",
                "summarizer": "execute_summarizer",
                "llm": "execute_llm",
                "done": "format_response"
            }
        )
        
        # Add edges from each tool execution to next step
        graph.add_edge("execute_calculator", "next_step")
        graph.add_edge("execute_code", "next_step")
        graph.add_edge("execute_web_search", "next_step")
        graph.add_edge("execute_summarizer", "next_step")
        graph.add_edge("execute_llm", "next_step")
        
        # Add conditional edge from next step back to routing
        graph.add_conditional_edges(
            "next_step",
            self._route_to_tool,
            {
                "calculator": "execute_calculator",
                "code_executor": "execute_code",
                "web_search": "execute_web_search",
                "summarizer": "execute_summarizer",
                "llm": "execute_llm",
                "done": "format_response"
            }
        )
        
        # Add edge from format response to end
        graph.add_edge("format_response", END)
        
        # Compile the graph
        compiled_graph = graph.compile()

        return compiled_graph
        
    # Synchronous wrapper methods for LangGraph nodes
    
    def _task_planner_node_sync(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for task planner node."""
        self.logger.log_step_start(state.get('current_step_index', 0) + 1, "TaskPlannerAgent")
        loop = asyncio.new_event_loop()
        try:
            result_state = loop.run_until_complete(self._task_planner_node(state))
        finally:
            loop.close()
        # Log the identified subgoals/steps
        steps = result_state.get('steps', [])
        if steps:
            subgoals = []
            for i, step in enumerate(steps, 1):
                tool = step.get('tool', 'llm')
                input_val = step.get('input', '')
                subgoals.append(f"{i}. {tool.title().replace('_', ' ')}: \"{input_val}\"")
            subgoals_str = "\n".join(subgoals)
            self.logger.log(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - [TaskPlannerAgent Output] - Identified subgoals:\n{subgoals_str}")
        return result_state
            
    def _calculator_node_sync(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for calculator node."""
        self.logger.log_step_start(state.get('current_step_index', 0) + 1, "CalculatorAgent")
        current_step = state["steps"][state["current_step_index"]]
        calc_state = {"expression": current_step.get("input", "")}
        if self.logger:
            self.logger.log_tool("calculator", f"Step {state['current_step_index']+1} input: {calc_state['expression']}")
        loop = asyncio.new_event_loop()
        try:
            result_state = loop.run_until_complete(self._calculator_node(state))
        finally:
            loop.close()
        if self.logger:
            # After async call, current_step is updated
            current_step = result_state["steps"][result_state["current_step_index"]]
            self.logger.log_tool("calculator", f"Step {result_state['current_step_index']+1} output: {current_step.get('result', '')}")
        return result_state
    
    def _code_executor_node_sync(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for code executor node."""
        self.logger.log_step_start(state.get('current_step_index', 0) + 1, "CodeExecutorAgent")
        current_step = state["steps"][state["current_step_index"]]
        code_state = {"code": current_step.get("input", "")}
        if self.logger:
            self.logger.log_tool("code_executor", f"Step {state['current_step_index']+1} input: {code_state['code']}")
        loop = asyncio.new_event_loop()
        try:
            result_state = loop.run_until_complete(self._code_executor_node(state))
        finally:
            loop.close()
        if self.logger:
            current_step = result_state["steps"][result_state["current_step_index"]]
            self.logger.log_tool("code_executor", f"Step {result_state['current_step_index']+1} output: {current_step.get('result', '')}")
        return result_state
    
    def _web_search_node_sync(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for web search node."""
        self.logger.log_step_start(state.get('current_step_index', 0) + 1, "WebSearchAgent")
        current_step = state["steps"][state["current_step_index"]]
        search_state = {"query": current_step.get("input", "")}
        if self.logger:
            self.logger.log_tool("web_search", f"Step {state['current_step_index']+1} input: {search_state['query']}")
        loop = asyncio.new_event_loop()
        try:
            result_state = loop.run_until_complete(self._web_search_node(state))
        finally:
            loop.close()
        if self.logger:
            current_step = result_state["steps"][result_state["current_step_index"]]
            self.logger.log_tool("web_search", f"Step {result_state['current_step_index']+1} output: {current_step.get('result', '')}")
        return result_state
    
    def _summarizer_node_sync(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for summarizer node."""
        self.logger.log_step_start(state.get('current_step_index', 0) + 1, "SummarizerAgent")
        current_step = state["steps"][state["current_step_index"]]
        summarizer_state = {"document": state.get("document", "")}
        if self.logger:
            self.logger.log_tool("summarizer", f"Step {state['current_step_index']+1} input: {summarizer_state['document']}")
        loop = asyncio.new_event_loop()
        try:
            result_state = loop.run_until_complete(self._summarizer_node(state))
        finally:
            loop.close()
        if self.logger:
            current_step = result_state["steps"][result_state["current_step_index"]]
            self.logger.log_tool("summarizer", f"Step {result_state['current_step_index']+1} output: {current_step.get('result', '')}")
        return result_state
    
    def _llm_node_sync(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for LLM node."""
        self.logger.log_step_start(state.get('current_step_index', 0) + 1, "LLMAgent")
        current_step = state["steps"][state["current_step_index"]]
        llm_state = {
            "query": current_step.get("input", ""),
            "chat_history": state.get("chat_history", []),
            "rag_memory": self.rag_memory
        }
        if current_step.get('rag_qa'):
            llm_state["rag_memory"] = self.rag_memory
        if hasattr(self, 'long_memory') and self.long_memory:
            llm_state["long_term_context"] = self.long_memory.get_all_facts()
        if self.rag_memory:
            try:
                relevant_docs = self.rag_memory.retrieve(current_step.get("input", ""), k=3)
                if relevant_docs:
                    llm_state["context"] = "\n\n".join(relevant_docs)
            except Exception as e:
                print(f"Error retrieving from RAG memory: {e}")
        if self.logger:
            self.logger.log_tool("llm", f"Step {state['current_step_index']+1} input: {llm_state['query']}")
        loop = asyncio.new_event_loop()
        try:
            result_state = loop.run_until_complete(self._llm_node(state))
        finally:
            loop.close()
        if self.logger:
            current_step = result_state["steps"][result_state["current_step_index"]]
            self.logger.log_tool("llm", f"Step {result_state['current_step_index']+1} output: {current_step.get('result', '')}")
        return result_state
            
    def _next_step_node_sync(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for next step node."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._next_step_node(state))
        finally:
            loop.close()
            
    def _format_response_node_sync(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for format response node."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._format_response_node(state))
        finally:
            loop.close()
    
    # Original async methods
    
    async def _task_planner_node(self, state: AgentState) -> AgentState:
        """Plan the tasks based on the user query."""
        planner_state = {
            "query": state["query"],
            "document": state["document"]
        }
        
        result_state = await self.task_planner.run(planner_state)
        
        # Update the state with steps
        if "steps" in result_state and result_state["steps"]:
            state["steps"] = [
                {
                    "tool": step.get("tool", "llm"),
                    "input": step.get("input", ""),
                    "status": "pending",
                    "result": None
                }
                for step in result_state["steps"]
            ]
        else:
            # Fallback to single LLM step
            state["steps"] = [
                {
                    "tool": "llm",
                    "input": state["query"],
                    "status": "pending",
                    "result": None
                }
            ]
        
        state["current_step_index"] = 0
        return state
    
    def _route_to_tool(self, state: AgentState) -> str:
        """Route to the appropriate tool based on the current step."""
        # Check if we're done with all steps
        if state["current_step_index"] >= len(state["steps"]):
            return "done"
        
        # Get the current step
        current_step = state["steps"][state["current_step_index"]]
        tool_name = current_step.get("tool", "llm")
        
        # Return the tool name
        if tool_name == "calculator":
            return "calculator"
        elif tool_name == "code_executor":
            return "code_executor"
        elif tool_name == "web_search":
            return "web_search"
        elif tool_name == "summarizer":
            return "summarizer"
        else:
            # Default to LLM
            return "llm"
    
    async def _calculator_node(self, state: AgentState) -> AgentState:
        """Execute the calculator tool."""
        current_step = state["steps"][state["current_step_index"]]
        calc_state = {"expression": current_step.get("input", "")}
        logger = state.get("logger")
        if logger:
            logger.log_tool("calculator", f"Step {state['current_step_index']+1} input: {calc_state['expression']}")
        result_state = await self.calculator.run(calc_state)
        if logger:
            logger.log_tool("calculator", f"Step {state['current_step_index']+1} output: {result_state.get('result', '')}")
        
        # Update the step with the result
        current_step["result"] = result_state.get("result", "Calculator error")
        current_step["status"] = "completed" if result_state.get("success", False) else "failed"
        
        # Add to results and failed steps if needed
        state["results"].append(current_step["result"])
        
        if not result_state.get("success", False):
            state["failed_steps"].append({
                "step": state["current_step_index"] + 1,
                "tool": "calculator",
                "input": current_step.get("input", ""),
                "error": current_step["result"]
            })
        
        # Update short-term and long-term memory
        if hasattr(self, 'memory') and self.memory:
            self.memory.add(current_step.get("input", ""), current_step["result"])
        if hasattr(self, 'long_memory') and self.long_memory:
            self.long_memory.add_fact({"input": current_step.get("input", ""), "result": current_step["result"]})
        
        # Log the step
        if logger:
            logger.log_step(state["current_step_index"] + 1, "CalculatorAgent", calc_state["expression"], current_step["result"], current_step["status"])
        
        return state
    
    async def _code_executor_node(self, state: AgentState) -> AgentState:
        """Execute the code executor tool."""
        current_step = state["steps"][state["current_step_index"]]
        code_state = {"code": current_step.get("input", "")}
        logger = state.get("logger")
        if logger:
            logger.log_tool("code_executor", f"Step {state['current_step_index']+1} input: {code_state['code']}")
        result_state = await self.code_executor.run(code_state)
        if logger:
            logger.log_tool("code_executor", f"Step {state['current_step_index']+1} output: {result_state.get('result', '')}")
        
        # Update the step with the result
        current_step["result"] = result_state.get('result', 'Code execution error')
        current_step["status"] = "completed" if result_state.get("success", False) else "failed"
        
        # Add to results and failed steps if needed
        state["results"].append(current_step["result"])
        
        if not result_state.get("success", False):
            state["failed_steps"].append({
                "step": state["current_step_index"] + 1,
                "tool": "code_executor",
                "input": current_step.get("input", ""),
                "error": current_step["result"]
            })
        
        # Log the step
        if logger:
            logger.log_step(state["current_step_index"] + 1, "CodeExecutorAgent", code_state["code"], current_step["result"], current_step["status"])
        
        return state
    
    async def _web_search_node(self, state: AgentState) -> AgentState:
        """Execute the web search tool."""
        current_step = state["steps"][state["current_step_index"]]
        search_state = {"query": current_step.get("input", "")}
        logger = state.get("logger")
        if logger:
            logger.log_tool("web_search", f"Step {state['current_step_index']+1} input: {search_state['query']}")
        result_state = await self.web_search.run(search_state)
        if logger:
            logger.log_tool("web_search", f"Step {state['current_step_index']+1} output: {result_state.get('result', '')}")
        
        # Update the step with the result
        current_step["result"] = result_state.get("result", "Web search error")
        current_step["status"] = "completed" if result_state.get("success", False) else "failed"
        
        # Add to results and failed steps if needed
        state["results"].append(current_step["result"])
        
        if not result_state.get("success", False):
            state["failed_steps"].append({
                "step": state["current_step_index"] + 1,
                "tool": "web_search",
                "input": current_step.get("input", ""),
                "error": current_step["result"]
            })
        
        # Log the step
        if logger:
            logger.log_step(state["current_step_index"] + 1, "WebSearchAgent", search_state["query"], current_step["result"], current_step["status"])
        
        return state
    
    async def _summarizer_node(self, state: AgentState) -> AgentState:
        """Execute the summarizer tool."""
        current_step = state["steps"][state["current_step_index"]]
        summarizer_state = {"document": state.get("document", "")}
        logger = state.get("logger")
        if logger:
            logger.log_tool("summarizer", f"Step {state['current_step_index']+1} input: {summarizer_state['document']}")
        result_state = await self.summarizer.run(summarizer_state)
        if logger:
            logger.log_tool("summarizer", f"Step {state['current_step_index']+1} output: {result_state.get('result', '')}")
        
        # Update the step with the result
        current_step["result"] = result_state.get("result", "Summarization error")
        current_step["status"] = "completed" if result_state.get("success", False) else "failed"
        
        # Add to results and failed steps if needed
        state["results"].append(current_step["result"])
        
        if not result_state.get("success", False):
            state["failed_steps"].append({
                "step": state["current_step_index"] + 1,
                "tool": "summarizer",
                "input": current_step.get("input", ""),
                "error": current_step["result"]
            })
        
        # Log the step
        if logger:
            logger.log_step(state["current_step_index"] + 1, "SummarizerAgent", summarizer_state["document"], current_step["result"], current_step["status"])
        
        return state
    
    async def _llm_node(self, state: AgentState) -> AgentState:
        """Execute the LLM tool."""
        current_step = state["steps"][state["current_step_index"]]
        llm_state = {
            "query": current_step.get("input", ""),
            "chat_history": state.get("chat_history", []),
            "rag_memory": self.rag_memory
        }
        if current_step.get('rag_qa'):
            llm_state["rag_memory"] = self.rag_memory
        if hasattr(self, 'long_memory') and self.long_memory:
            llm_state["long_term_context"] = self.long_memory.get_all_facts()
        if self.rag_memory:
            try:
                relevant_docs = self.rag_memory.retrieve(current_step.get("input", ""), k=3)
                if relevant_docs:
                    llm_state["context"] = "\n\n".join(relevant_docs)
            except Exception as e:
                print(f"Error retrieving from RAG memory: {e}")
        logger = state.get("logger")
        if logger:
            logger.log_tool("llm", f"Step {state['current_step_index']+1} input: {llm_state['query']}")
        result_state = await self.llm.run(llm_state)
        if logger:
            logger.log_tool("llm", f"Step {state['current_step_index']+1} output: {result_state.get('result', '')}")
        
        # Update the step with the result
        current_step["result"] = result_state.get("result", "LLM error")
        current_step["status"] = "completed" if result_state.get("success", False) else "failed"
        
        # Add to results and failed steps if needed
        state["results"].append(current_step["result"])
        
        if not result_state.get("success", False):
            state["failed_steps"].append({
                "step": state["current_step_index"] + 1,
                "tool": "llm",
                "input": current_step.get("input", ""),
                "error": current_step["result"]
            })
        
        # Update short-term and long-term memory
        if hasattr(self, 'memory') and self.memory:
            self.memory.add(current_step.get("input", ""), current_step["result"])
        if hasattr(self, 'long_memory') and self.long_memory:
            self.long_memory.add_fact({"input": current_step.get("input", ""), "result": current_step["result"]})
        
        # Log the step
        if logger:
            logger.log_step(state["current_step_index"] + 1, "LLMAgent", llm_state["query"], current_step["result"], current_step["status"])
        
        return state
    
    async def _next_step_node(self, state: AgentState) -> AgentState:
        """Move to the next step."""
        state["current_step_index"] += 1
        return state
    
    async def _format_response_node(self, state: AgentState) -> AgentState:
        """Format the final response."""
        original_query = state["query"]
        results = state["results"]
        failed_steps = state["failed_steps"]
        total_steps = len(state["steps"])

        # Map tool names to pretty agent headers
        tool_headers = {
            "web_search": "## Web Search Agent:",
            "calculator": "## Calculator Agent:",
            "summarizer": "## Summarizer Agent:",
            "code_executor": "## Code Executor Agent:",
            "llm": "## LLM Agent:"
        }
        formatted_results = []
        for i, step in enumerate(state["steps"]):
            tool = step.get("tool", "llm")
            result = step.get("result", "")
            header = tool_headers.get(tool, f"## {tool.title()} Agent:")
            formatted_results.append(f"{header}\n{result}")

        if not failed_steps:
            state["final_response"] = "\n\n".join(formatted_results) if len(formatted_results) > 1 else (formatted_results[0] if formatted_results else "")
            return state

        successful_count = total_steps - len(failed_steps)
        response_parts = []

        if successful_count == 0:
            response_parts.append(f'❌ **All steps failed** for your request: "{original_query}"')
        else:
            response_parts.append(f"✅ **Partial Success:** {successful_count} out of {total_steps} steps completed.")
            response_parts.append("\n**✅ Successful Results:**")
            response_parts.extend(formatted_results)

        response_parts.append("\n**❌ Failed Steps:**")
        for fs in failed_steps:
            response_parts.append(f"• Step {fs['step']} ({fs['tool']}): {fs['error']}")
        state["final_response"] = "\n".join(response_parts)
        return state

    def _log_step(self, step_num, agent_name, input_text, output_text):
        self.logger.log_agent(agent_name, f"Step {step_num} input: {input_text}")
        self.logger.log_agent(agent_name, f"Step {step_num} output: {output_text}")

    def _format_final_response(self, state: dict) -> dict:
        """Formats the final response by combining results from all steps into a markdown string."""
        final_response = "### ✅ All steps completed successfully!\n\n"
        all_results = ""

        # Iterate through the history of agent results
        for i, step_result in enumerate(state.get("agent_results", [])):
            agent_name = step_result.get("agent", "Unknown Agent")
            output = step_result.get("output", "No output.")
            
            # Sanitize output for markdown
            if isinstance(output, str):
                output = output.replace('$', '\\$') # Escape dollar signs for markdown
            else:
                output = f"```\n{output}\n```"

            all_results += f"### {agent_name} Results:\n{output}\n\n---\n\n"

        if not all_results:
            final_response = "No results were generated."
        else:
            final_response += all_results
            
        return {"final_response": final_response}

if __name__ == "__main__":
    agent = MasterAgent()
    graph = agent.graph
    try:
        mermaid = graph.get_graph().draw_mermaid()
        with open("langgraph_graph.mmd", "w") as f:
            f.write(mermaid)
        ascii_graph = graph.get_graph().draw_ascii()
        with open("langgraph_graph.txt", "w") as f:
            f.write(ascii_graph)
        print("LangGraph graph exported to langgraph_graph.mmd (Mermaid) and langgraph_graph.txt (ASCII)")
    except Exception as e:
        print(f"[WARN] Could not export LangGraph graph: {e}") 