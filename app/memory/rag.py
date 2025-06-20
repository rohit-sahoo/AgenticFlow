import os
from typing import List, Dict, Any, Optional
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGMemory:
    def __init__(self, persist_dir: str = "data/rag_memory"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("rag_memory")
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None):
        if not text or not text.strip():
            return
        meta = meta or {}
        chunks = self.text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                embedding = self.embedder.embed_documents([chunk])[0]
                chunk_id = f"{hash(text)}_{i}"
                chunk_meta = meta.copy()
                chunk_meta.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                })
                try:
                    self.collection.add(
                        documents=[chunk], 
                        metadatas=[chunk_meta], 
                        embeddings=[embedding], 
                        ids=[chunk_id]
                    )
                except Exception as e:
                    print(f"Error adding chunk to RAG memory: {e}")

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if not query or not query.strip():
            return []
        try:
            embedding = self.embedder.embed_query(query)
            results = self.collection.query(
                query_embeddings=[embedding], 
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            docs = results.get("documents", [])
            if docs and len(docs) > 0:
                return docs[0]
            return []
        except Exception as e:
            print(f"Error retrieving from RAG memory: {e}")
            return []

    def clear(self):
        try:
            self.client.delete_collection("rag_memory")
            self.collection = self.client.get_or_create_collection("rag_memory")
        except Exception as e:
            print(f"Error clearing RAG memory: {e}")

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": "rag_memory"
            }
        except Exception as e:
            return {"error": str(e)} 