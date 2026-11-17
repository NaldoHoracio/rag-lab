"""
@author: Edvonaldo Horacio (edvonaldohoracio@gmail.com)
@date: 16-11-2026
@description: Faz perguntas ao RAG pelo terminal: busca os trechos mais
              relevantes no ChromaDB e gera a resposta com o LLM local.

Uso (a partir da raiz do repositório):
    python src/ask.py "Quantos canais de ADC o ESP32 tem?"
    python src/ask.py                      # modo interativo
    python src/ask.py "..." --k 8          # usa mais trechos de contexto
    python src/ask.py "..." --show-context # mostra os trechos enviados ao LLM
    python src/ask.py "..." --no-llm       # só a busca, sem gerar resposta
"""

from __future__ import annotations

import argparse
import sys

from config import settings
from document_loader import Chunk
from generator import OllamaError, OllamaGenerator, build_messages


def retrieve(question: str, store, embedder, k: int) -> list[tuple[Chunk, float]]:
    """Etapa de retrieval: transforma a pergunta em vetor e busca os k trechos mais parecidos."""
    query_vector = embedder.encode([question])[0]
    return store.search(query_vector, k)


def print_sources(hits: list[tuple[Chunk, float]]) -> None:
    print("\n📚 Fontes:")
    for i, (chunk, score) in enumerate(hits, start=1):
        print(f"  [{i}] {chunk.source}, pág. {chunk.page}  (similaridade {score:.2f})")


def print_context(hits: list[tuple[Chunk, float]]) -> None:
    print("\n🔎 Trechos enviados ao LLM:")
    for i, (chunk, score) in enumerate(hits, start=1):
        print(f"\n--- [{i}] {chunk.source}, pág. {chunk.page} ({score:.2f}) ---")
        print(chunk.text)


def answer(question: str, store, embedder, generator, k: int, show_context: bool, no_llm: bool) -> None:
    hits = retrieve(question, store, embedder, k)
    if not hits:
        print("Nenhum trecho encontrado. Rode antes: python src/build_index.py")
        return

    if show_context or no_llm:
        print_context(hits)
    if not no_llm:
        print("\n💬 Resposta:\n")
        for piece in generator.stream(build_messages(question, hits)):
            print(piece, end="", flush=True)
        print()
    print_sources(hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pergunte algo sobre os documentos indexados.")
    parser.add_argument("question", nargs="?", help="pergunta (sem ela, abre o modo interativo)")
    parser.add_argument("--k", type=int, default=settings.top_k, help="número de trechos de contexto")
    parser.add_argument("--model", default=settings.llm_model, help="modelo do Ollama")
    parser.add_argument("--show-context", action="store_true", help="mostra os trechos recuperados")
    parser.add_argument("--no-llm", action="store_true", help="só faz a busca, sem gerar resposta")
    args = parser.parse_args()

    from embeddings import get_embedder
    from vector_store import ChromaStore

    embedder = get_embedder()
    store = ChromaStore(settings.persist_directory, settings.collection_name, embedder.name)
    if store.count() == 0:
        sys.exit("O banco está vazio. Rode antes: python src/build_index.py")

    generator = OllamaGenerator(
        settings.ollama_url, args.model, settings.llm_temperature, settings.llm_timeout
    )
    run = lambda q: answer(q, store, embedder, generator, args.k, args.show_context, args.no_llm)  # noqa: E731

    try:
        if args.question:
            run(args.question)
            return

        print(f"🤖 rag-lab ({store.count()} chunks, LLM: {args.model}). Digite 'sair' para encerrar.")
        while True:
            try:
                question = input("\n❓ Pergunta: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"", "sair", "exit", "quit"}:
                break
            run(question)
    except OllamaError as e:
        sys.exit(f"\n❌ {e}")


if __name__ == "__main__":
    main()
