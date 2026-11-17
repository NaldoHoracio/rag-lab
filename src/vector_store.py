"""
@author: Edvonaldo Horacio (edvonaldohoracio@gmail.com)
@date: 16-11-2026
@description: Banco vetorial com ChromaDB persistente.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from document_loader import Chunk

# O Chroma limita o tamanho de cada lote de inserção
MAX_BATCH = 1000


class ChromaStore:
    def __init__(self, persist_directory: Path, collection_name: str, embedding_model: str | None = None):
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        metadata = {"hnsw:space": "cosine"}  # similaridade de cosseno
        if embedding_model:
            metadata["embedding_model"] = embedding_model
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata=metadata)

        # Vetores de modelos diferentes não são comparáveis: avisa se a coleção
        # foi criada com outro modelo de embeddings.
        stored = (self.collection.metadata or {}).get("embedding_model")
        if embedding_model and stored and stored != embedding_model:
            raise ValueError(
                f"A coleção '{collection_name}' foi indexada com '{stored}', "
                f"mas o modelo atual é '{embedding_model}'. "
                "Reindexe com: python src/build_index.py --reset"
            )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insere (ou atualiza) chunks. Como o ID é estável (arquivo:página:índice),
        reindexar o mesmo PDF não duplica dados."""
        if len(chunks) != len(embeddings):
            raise ValueError("número de chunks e de embeddings não confere")
        for i in range(0, len(chunks), MAX_BATCH):
            batch = chunks[i : i + MAX_BATCH]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                embeddings=embeddings[i : i + MAX_BATCH],
            )

    def search(self, query_embedding: list[float], k: int) -> list[tuple[Chunk, float]]:
        """Devolve os k chunks mais parecidos com a pergunta e sua similaridade."""
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunk = Chunk(text=doc, source=meta["source"], page=meta["page"], chunk_id=meta["chunk_id"])
            hits.append((chunk, 1.0 - dist))  # distância de cosseno -> similaridade
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Apaga e recria a coleção (útil ao mudar o modelo de embeddings)."""
        name, metadata = self.collection.name, self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name, metadata=metadata)

    def sources(self) -> set[str]:
        """Nomes dos arquivos já indexados."""
        metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        return {m["source"] for m in metadatas}
