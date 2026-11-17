"""
@author: Edvonaldo Horacio (edvonaldohoracio@gmail.com)
@date: 16-11-2026
@description: Configurações centrais do projeto. Todos os valores podem ser
              sobrescritos por variáveis de ambiente ou pelo arquivo .env.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class Settings:
    # Caminhos
    data_path: Path = Path(os.getenv("RAG_DATA_PATH", ROOT_DIR / "data" / "data-raw"))
    persist_directory: Path = Path(os.getenv("RAG_PERSIST_DIR", ROOT_DIR / "chroma_db"))
    collection_name: str = os.getenv("RAG_COLLECTION", "rag_lab")

    # Chunking (em caracteres)
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", 200))

    # Embeddings: "huggingface" (local, gratuito) ou "openai" (requer OPENAI_API_KEY)
    embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "huggingface")
    openai_embedding_model: str = os.getenv("RAG_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    hf_embedding_model: str = os.getenv(
        "RAG_HF_EMBEDDING_MODEL",
        # multilíngue: pergunta em PT, documento em EN
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", 5))

    # Geração (Ollama local)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "llama3.2:3b")
    llm_temperature: float = float(os.getenv("RAG_LLM_TEMPERATURE", 0.1))
    llm_timeout: int = int(os.getenv("RAG_LLM_TIMEOUT", 300))  # segundos


settings = Settings()
