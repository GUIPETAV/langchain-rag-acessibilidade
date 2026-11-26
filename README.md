# langchain-rag-acessibilidade

Assistente didático de **perguntas e respostas sobre acessibilidade digital**, construído como pipeline **RAG com LangChain** em Python. A base de conhecimento cobre WCAG, texto alternativo, contraste, teclado, leitores de tela, legendas e Libras, linguagem simples, legislação brasileira (LBI e eMAG) e o uso de IA em acessibilidade. Roda **100% offline** (sem chave de API) e aceita um LLM opcional para gerar a resposta final.

## O que o projeto mostra

1. **Ingestão e chunking** de documentos Markdown com `RecursiveCharacterTextSplitter` (divisão por seção).
2. **Busca híbrida**: `BM25Retriever` (lexical) + `TFIDFRetriever` (vetorial esparso), com pré-processamento e stopwords em português.
3. **Fusão por RRF** (Reciprocal Rank Fusion) dos dois rankings.
4. **Prompt com citação de fontes** (`[n]`) e instrução de não responder fora do contexto.
5. **Geração**: com `OPENAI_API_KEY` definida usa `ChatOpenAI`; sem chave, devolve o trecho mais relevante (modo extrativo).
6. **Avaliação do retrieval** (`hit-rate@k`) e testes automatizados com `pytest`.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m rag "Qual o contraste mínimo de cores?"
PYTHONPATH=src python -m rag.evaluate
pytest
```

Resultado da avaliação no conjunto de exemplo (7 perguntas): `hit-rate@4 = 1.00` e `hit-rate@1 = 1.00`. O conjunto é pequeno e foi escrito junto com a base, então o resultado mostra que o pipeline funciona, não que ele generaliza.

Para usar um LLM: `pip install langchain-openai` e defina `OPENAI_API_KEY` (e, opcionalmente, `OPENAI_MODEL`).

## Limitações (de propósito, por ser didático)

- Base de exemplo pequena (`data/acessibilidade_digital.md`), escrita como resumo didático e **não** substitui o texto oficial do WCAG ou da legislação e conjunto de avaliação com poucas perguntas.
- Os vetores são TF-IDF, não embeddings densos; o próximo passo natural é trocar por embeddings e um vector DB (pgvector, Chroma ou Qdrant) mantendo a fusão RRF.
- Sem reranking, sem observabilidade e sem implantação em produção.

## Estrutura

```
data/                  base de conhecimento sobre acessibilidade digital
src/rag/pipeline.py    chunking, retrievers, RRF, prompt e resposta
src/rag/evaluate.py    hit-rate@k
tests/                 testes com pytest
```

## Sobre a acessibilidade do próprio projeto

A interface é um comando de terminal com saída em texto puro, sem cores nem elementos gráficos, o que funciona com leitores de tela. Ainda não há interface web; ao criar uma, ela deve seguir as práticas descritas na base de conhecimento (HTML semântico, navegação por teclado, contraste e rótulos).
