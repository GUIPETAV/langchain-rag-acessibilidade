"""Pipeline RAG didático com LangChain: chunking -> busca híbrida (BM25 + TF-IDF) -> RRF -> resposta."""
from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


STOPWORDS = {
    "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "e", "é", "que", "para", "por", "com", "se", "ao", "como", "posso", "pode", "devem", "deve",
    "ser", "sao", "são", "quais", "qual", "ter", "tem", "ou", "ma", "me", "meu", "sua", "seu",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def load_documents(data_dir: str | Path) -> list[Document]:
    docs = []
    for path in sorted(Path(data_dir).glob("*.md")):
        docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}))
    return docs


def chunk(docs: list[Document], size: int = 300, overlap: int = 0) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n## ", "\n\n", "\n", " "]
    )
    return splitter.split_documents(docs)


def rrf(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: combina rankings de vários retrievers."""
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for ranking in rankings:
        for pos, doc in enumerate(ranking):
            key = doc.page_content
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + pos + 1)
    return [by_key[key] for key in sorted(scores, key=scores.get, reverse=True)]


class HybridRetriever:
    def __init__(self, chunks: list[Document], top_k: int = 4):
        self.top_k = top_k
        self.bm25 = BM25Retriever.from_documents(chunks, k=top_k * 2, preprocess_func=tokenize)
        self.tfidf = TFIDFRetriever.from_documents(
            chunks, k=top_k * 2, tfidf_params={"tokenizer": tokenize, "lowercase": False, "token_pattern": None}
        )

    def retrieve(self, query: str) -> list[Document]:
        fused = rrf([self.bm25.invoke(query), self.tfidf.invoke(query)])
        return fused[: self.top_k]


def build_prompt(question: str, contexts: list[Document]) -> str:
    blocks = [f"[{i}] ({d.metadata.get('source')}) {d.page_content}" for i, d in enumerate(contexts, 1)]
    return (
        "Responda apenas com base no contexto. Cite as fontes como [n]. "
        "Se a resposta não estiver no contexto, diga que não sabe.\n\n"
        "Contexto:\n" + "\n\n".join(blocks) + f"\n\nPergunta: {question}\nResposta:"
    )


def answer(question: str, retriever: HybridRetriever) -> dict:
    contexts = retriever.retrieve(question)
    prompt = build_prompt(question, contexts)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI  # pip install langchain-openai

        text = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")).invoke(prompt).content
    else:  # modo offline: resposta extrativa (trecho mais relevante)
        text = f"(modo offline) {contexts[0].page_content.strip()} [1]" if contexts else "Não sei."
    return {"answer": text, "sources": [d.metadata.get("source") for d in contexts], "prompt": prompt}
