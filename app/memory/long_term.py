import os
import json
from typing import List, Dict, Any

class LongTermMemory:
    def __init__(self, path: str = "data/long_term_memory.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def add_fact(self, fact: Dict[str, Any]):
        facts = self.get_all_facts()
        facts.append(fact)
        with open(self.path, "w") as f:
            json.dump(facts, f, indent=2)

    def get_all_facts(self) -> List[Dict[str, Any]]:
        with open(self.path, "r") as f:
            return json.load(f)

    def search_facts(self, query: str) -> List[Dict[str, Any]]:
        facts = self.get_all_facts()
        return [fact for fact in facts if query.lower() in json.dumps(fact).lower()] 