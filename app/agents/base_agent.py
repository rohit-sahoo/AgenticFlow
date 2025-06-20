"""
Base agent class that defines the common interface for all agents.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str):
        """Initialize the agent with a name."""
        self.name = name
    
    @abstractmethod
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the given state.
        
        Args:
            state: The current state of the workflow
            
        Returns:
            Updated state after the agent has processed it
        """
        pass
    
    def get_name(self) -> str:
        """Return the name of the agent."""
        return self.name 