"""memory.py — ChromaDB RAG for cross-session memory (Week 4)."""

class MemoryStore:
    def add(self, session_id: str, text: str):
        raise NotImplementedError

    def query(self, text: str, n: int = 5) -> list:
        raise NotImplementedError
