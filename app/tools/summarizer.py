class DocumentSummarizerTool:
    """A tool to summarize documents using the LLM with enhanced error handling."""
    def __init__(self, llm):
        self.llm = llm
        
    def run(self, document: str) -> str:
        if not document:
            return "No document is currently available to summarize. Please provide a document first."
        stripped_doc = document.strip()
        if not stripped_doc:
            return "No document is currently available to summarize. Please provide a document first."
        if len(stripped_doc) < 10:
            return "No document is currently available to summarize. Please provide a document first."
        summarizer_llm = self.llm
        prompt = (
            "CRITICAL INSTRUCTION: You are a document summarizer. You must ONLY summarize the document provided below. "
            "If no document is provided or if the document is empty, respond with: 'No document is currently available to summarize. Please provide a document first.' "
            "Do NOT reference any other documents, conversations, or external knowledge. "
            "Do NOT make up content. Only summarize what is explicitly provided.\n\n"
            "Document to summarize:\n" + stripped_doc + "\n\n"
            "If the above document is empty or contains no meaningful content, respond with: 'No document is currently available to summarize. Please provide a document first.' "
            "Otherwise, provide a 5-10 sentence summary focusing on the main points and key details."
        )
        try:
            response = summarizer_llm.invoke([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, 'content') else str(response)
            result = str(content)
            if "no document" in result.lower() and len(stripped_doc) > 10:
                retry_prompt = f"Summarize this document in 5-10 sentences:\n\n{stripped_doc}"
                response = summarizer_llm.invoke([{"role": "user", "content": retry_prompt}])
                content = response.content if hasattr(response, 'content') else str(response)
                result = str(content)
            return result
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and "rate limit" in error_str.lower():
                import re
                wait_match = re.search(r'try again in (\d+\.?\d*)s', error_str)
                wait_time = float(wait_match.group(1)) if wait_match else 20.0
                return f"Rate limit exceeded. Please try again in {wait_time:.1f} seconds."
            else:
                return f"Error summarizing document: {e}" 