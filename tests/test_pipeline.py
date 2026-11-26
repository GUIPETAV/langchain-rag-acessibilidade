from pathlib import Path

from rag.evaluate import hit_rate
from rag.pipeline import HybridRetriever, answer, chunk, load_documents, rrf
from langchain_core.documents import Document

DATA = Path(__file__).resolve().parents[1] / "data"


def make():
    return HybridRetriever(chunk(load_documents(DATA)))


def test_chunking_creates_multiple_chunks():
    assert len(chunk(load_documents(DATA))) > 3


def test_rrf_prefers_docs_in_both_rankings():
    a, b, c = Document(page_content="a"), Document(page_content="b"), Document(page_content="c")
    assert rrf([[a, b], [b, c]])[0].page_content == "b"


def test_hit_rate_is_high():
    assert hit_rate(make()) >= 0.8


def test_answer_returns_sources():
    out = answer("O que é o eMAG?", make())
    assert out["sources"] and "Pergunta: O que é o eMAG?" in out["prompt"]


def test_top1_hit_rate():
    assert hit_rate(make(), top1=True) >= 0.8
