"""
@author: Edvonaldo Horacio (edvonaldohoracio@gmail.com)
@date: 16-11-2026
@description: Indexa os PDFs de data/data-raw no ChromaDB.

Uso (a partir da raiz do repositório):
    python src/build_index.py              # indexa todos os PDFs
    python src/build_index.py --dry-run    # só extrai e divide, sem gerar embeddings
    python src/build_index.py --reset      # apaga a coleção e indexa do zero
    python src/build_index.py --skip-existing   # pula PDFs já indexados
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from config import settings
from document_loader import chunk_pdf

# Páginas com menos texto que isso provavelmente são imagens (PDF escaneado)
MIN_CHARS_PER_PAGE = 50


def find_pdfs(data_path: Path) -> list[Path]:
    pdfs = sorted(Path(data_path).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Nenhum PDF encontrado em {data_path}")
    return pdfs


def build_index(
    pdfs: list[Path],
    store=None,
    embedder=None,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Processa cada PDF (extrai, divide, gera embeddings e salva) e devolve
    o número de chunks por arquivo. Cada arquivo é salvo assim que termina,
    então uma falha no meio não perde o que já foi indexado."""
    already = store.sources() if (store is not None and skip_existing) else set()
    summary: dict[str, int] = {}

    for i, pdf in enumerate(pdfs, start=1):
        prefix = f"[{i}/{len(pdfs)}] {pdf.name}"
        if pdf.name in already:
            print(f"{prefix}: já indexado, pulando")
            continue

        start = time.perf_counter()
        chunks = chunk_pdf(pdf, chunk_size, overlap)
        summary[pdf.name] = len(chunks)

        if not chunks:
            print(f"{prefix}: ⚠️ nenhum texto extraído (PDF escaneado?)")
            continue

        pages = {c.page for c in chunks}
        avg_chars = sum(len(c.text) for c in chunks) / len(pages)
        warning = " ⚠️ pouco texto por página" if avg_chars < MIN_CHARS_PER_PAGE else ""

        if dry_run:
            print(f"{prefix}: {len(pages)} páginas com texto, {len(chunks)} chunks{warning}")
            continue

        vectors = embedder.encode([c.text for c in chunks])
        store.add(chunks, vectors)
        elapsed = time.perf_counter() - start
        print(f"{prefix}: {len(pages)} páginas, {len(chunks)} chunks em {elapsed:.1f}s{warning}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa os PDFs no ChromaDB.")
    parser.add_argument("--data-path", type=Path, default=settings.data_path)
    parser.add_argument("--reset", action="store_true", help="apaga a coleção antes de indexar")
    parser.add_argument("--skip-existing", action="store_true", help="pula PDFs já indexados")
    parser.add_argument("--dry-run", action="store_true", help="não gera embeddings nem salva")
    parser.add_argument("--provider", choices=["openai", "huggingface"], help="sobrescreve o .env")
    args = parser.parse_args()

    pdfs = find_pdfs(args.data_path)
    print(f"📂 {len(pdfs)} PDF(s) em {args.data_path}")
    print(f"✂️  chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")

    store = embedder = None
    if not args.dry_run:
        from embeddings import get_embedder
        from vector_store import ChromaStore

        embedder = get_embedder(args.provider)
        print(f"🧠 Embeddings: {embedder.name}")
        if args.reset:
            # abre sem checar o modelo, já que a coleção vai ser recriada
            store = ChromaStore(settings.persist_directory, settings.collection_name)
            store.client.delete_collection(settings.collection_name)
            print("🗑️  Coleção apagada")
        store = ChromaStore(settings.persist_directory, settings.collection_name, embedder.name)

    start = time.perf_counter()
    summary = build_index(
        pdfs, store, embedder, skip_existing=args.skip_existing, dry_run=args.dry_run
    )
    total = sum(summary.values())
    print(f"\n✅ {total} chunks processados em {time.perf_counter() - start:.1f}s")
    if store is not None:
        print(f"💾 Coleção '{settings.collection_name}' em {settings.persist_directory}: {store.count()} chunks")


if __name__ == "__main__":
    main()
