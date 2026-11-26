import sys
from pathlib import Path

from rag.pipeline import HybridRetriever, answer, chunk, load_documents

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "O que é RAG?"
    data = Path(__file__).resolve().parents[2] / "data"
    result = answer(question, HybridRetriever(chunk(load_documents(data))))
    print(result["answer"])
    print("Fontes:", result["sources"])
