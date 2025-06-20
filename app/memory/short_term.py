class ShortTermMemory:
    def __init__(self):
        self.history = []

    def add(self, user, ai):
        self.history.append((user, ai))

    def get(self):
        return self.history

    def get_recent(self, n: int = 5) -> str:
        if not self.history:
            return ""
        recent = self.history[-n:] if len(self.history) > n else self.history
        formatted = []
        for user, ai in recent:
            ai_clean = ai.strip()
            if "Document Summary:" in ai_clean:
                summary_start = ai_clean.find("Document Summary:")
                if summary_start != -1:
                    summary_content = ai_clean[summary_start:].replace("Document Summary:", "").strip()
                    formatted.append(f"User: {user}")
                    formatted.append(f"AI: Document Summary - {summary_content}")
                else:
                    formatted.append(f"User: {user}")
                    formatted.append(f"AI: {ai_clean}")
            else:
                formatted.append(f"User: {user}")
                formatted.append(f"AI: {ai_clean}")
        return "\n".join(formatted)

    def clear(self):
        self.history = [] 