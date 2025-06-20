"""
main application with LangGraph orchestration.
"""
import gradio as gr
from gradio.themes import Soft
import asyncio
from dotenv import load_dotenv
import os
from app.memory import ShortTermMemory, LongTermMemory, RAGMemory
from app.agents.master_agent import MasterAgent

# Load environment variables
load_dotenv()

# Initialize memory systems
memory = ShortTermMemory()
long_memory = LongTermMemory()
rag_memory = RAGMemory()

# Clean the RAG vector DB at startup
rag_memory.clear()
print("[Startup] RAG vector DB cleared.")

# Initialize the master agent
master_agent = MasterAgent(memory=memory, rag_memory=rag_memory, long_memory=long_memory)

SUGGESTIONS = [
    ["Search for latest news on AI", None],
    ["Upload a PDF and ask: 'Summarize this document.'", None],
    ["Calculate: 42 * sqrt(19) + abs(-7) / 3", None],
    ["Run this code: print('Hello, world!')", None],
    ["What is the capital of France? (with RAG doc upload)", None]
]

def extract_text_from_file(file):
    """Extract text from various file formats."""
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith('.txt'):
        return file.read().decode('utf-8')
    elif name.endswith('.pdf'):
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        text = "\n".join(page.extract_text() or '' for page in reader.pages)
        return text
    elif name.endswith('.docx'):
        from docx import Document
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return None

async def chat_fn(message, history, file=None):
    if not history:  # New session
        memory.clear()
    document_text = ""
    if file:
        try:
            print(f"Processing uploaded file: {file.name}")
            document_text = extract_text_from_file(file)
            if document_text and document_text.strip():
                print(f"Adding document '{file.name}' to RAG memory...")
                rag_memory.add(document_text, {"type": "document", "filename": file.name})
                stats = rag_memory.get_stats()
                print(f"RAG memory now contains {stats.get('total_documents', 0)} chunks")
            else:
                print(f"Could not extract text from file: {file.name}")
        except Exception as e:
            print(f"Error processing file: {e}")
            return f"Error processing file: {e}"
    state = {
        "query": message,
        "document": document_text,
        "chat_history": history
    }
    try:
        result = await master_agent.run(state)
        return result.get("final_response", "No response generated.")
    except Exception as e:
        import traceback
        print(f"Error running master agent: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"

def sync_chat_fn(message, history, file=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(chat_fn(message, history, file))
    finally:
        loop.close()

demo = gr.ChatInterface(
    fn=sync_chat_fn,
    title="Multi Agent LangGraph App",
    description="A modular AI agent with LangGraph orchestration, featuring web search, document summarization, code execution, and calculations.",
    additional_inputs=[gr.File(label="Attach a file (PDF, DOCX, TXT)", file_types=[".pdf", ".docx", ".txt"])],
    examples=SUGGESTIONS,
    theme=Soft(),
    submit_btn="Send"
)

def main():
    demo.launch()

if __name__ == "__main__":
    main() 