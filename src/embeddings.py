"""
@author: Edvonaldo Horacio (edvonaldohoracio@gmail.com)
@date: 16-11-2026
@description: Geração de embeddings. Suporta a API da OpenAI e modelos locais
              do HuggingFace (sentence-transformers), escolhidos pelo config.
"""

from __future__ import annotations

import os

from config import ENV_FILE, settings


class OpenAIEmbedder:
    def __init__(self, model: str):
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            status = "existe, mas não define a variável" if ENV_FILE.exists() else "não foi encontrado"
            raise SystemExit(
                f"❌ OPENAI_API_KEY não definida. O arquivo {ENV_FILE} {status}.\n"
                "   Crie-o na raiz do repositório com a linha:\n"
                "   OPENAI_API_KEY=sk-..."
            )

        self.model = model
        self.name = f"openai/{model}"
        self.client = OpenAI()  # lê OPENAI_API_KEY do ambiente / .env

    def encode(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            response = self.client.embeddings.create(
                input=texts[i : i + batch_size], model=self.model
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors


class HuggingFaceEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.name = f"huggingface/{model}"
        self.model = SentenceTransformer(model)

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            normalize_embeddings=True,
        )
        return vectors.tolist()


def get_embedder(provider: str | None = None):
    """Devolve o embedder configurado em RAG_EMBEDDING_PROVIDER."""
    provider = (provider or settings.embedding_provider).lower()
    if provider == "openai":
        return OpenAIEmbedder(settings.openai_embedding_model)
    if provider == "huggingface":
        return HuggingFaceEmbedder(settings.hf_embedding_model)
    raise ValueError(f"Provedor de embeddings desconhecido: {provider}")
