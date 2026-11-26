"""Avaliação simples do retrieval: hit-rate@k em um conjunto de perguntas com trecho esperado."""
from __future__ import annotations

from pathlib import Path

from rag.pipeline import HybridRetriever, chunk, load_documents

GOLDEN = [
    ("Quais são os princípios do WCAG?", "POUR"),
    ("Como descrever imagens para leitor de tela?", "texto alternativo"),
    ("Qual o contraste mínimo de cores?", "4,5 para 1"),
    ("Dá para usar o site só com o teclado?", "ordem de tabulação"),
    ("O que é o eMAG?", "modelo de acessibilidade do governo"),
    ("Vídeos precisam de legendas?", "legendas"),
    ("A IA pode escrever o texto alternativo sozinha?", "revisão humana"),
]


def hit_rate(retriever: HybridRetriever, top1: bool = False) -> float:
    hits = 0
    for question, expected in GOLDEN:
        docs = retriever.retrieve(question)[: 1 if top1 else None]
        texts = " ".join(d.page_content for d in docs)
        hits += expected.lower() in texts.lower()
    return hits / len(GOLDEN)


if __name__ == "__main__":
    data = Path(__file__).resolve().parents[2] / "data"
    r = HybridRetriever(chunk(load_documents(data)))
    print(f"hit-rate@{r.top_k}: {hit_rate(r):.2f}")
    print(f"hit-rate@1: {hit_rate(r, top1=True):.2f}")
