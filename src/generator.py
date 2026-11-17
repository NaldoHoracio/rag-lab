"""
@author: Edvonaldo Horacio (edvonaldohoracio@gmail.com)
@date: 16-11-2026
@description: Etapa de geração do RAG. Monta o prompt com os trechos
              recuperados e envia ao LLM local via Ollama.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import requests

from document_loader import Chunk

SYSTEM_PROMPT = """Você é um assistente técnico especializado em microcontroladores e sistemas embarcados.
Responda à pergunta do usuário usando APENAS as informações dos trechos de documentação abaixo.

Regras:
- Responda sempre em português, mesmo que os trechos estejam em inglês.
- Mantenha no original os nomes de registradores, pinos, bits e siglas (ex.: TCCR1B, GPIO, ADC).
- Cite a origem de cada informação com o número do trecho entre colchetes, por exemplo [1] ou [2][3].
- Se os trechos não contiverem a resposta, diga claramente que não encontrou a informação
  nos documentos. Não use conhecimento próprio nem invente valores.
- Seja direto e objetivo."""


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    """Numera os trechos recuperados para que o LLM possa citá-los."""
    blocks = []
    for i, (chunk, _score) in enumerate(hits, start=1):
        blocks.append(f"[{i}] (arquivo: {chunk.source}, página {chunk.page})\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, hits: list[tuple[Chunk, float]]) -> list[dict]:
    user_content = f"Trechos da documentação:\n\n{format_context(hits)}\n\nPergunta: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class OllamaError(RuntimeError):
    """Erro com mensagem amigável para problemas de conexão ou de modelo."""


class OllamaGenerator:
    def __init__(self, url: str, model: str, temperature: float = 0.1, timeout: int = 300):
        self.url = url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Envia as mensagens ao Ollama e devolve a resposta aos pedaços,
        conforme o modelo vai gerando (streaming)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        try:
            response = requests.post(
                f"{self.url}/api/chat", json=payload, stream=True, timeout=self.timeout
            )
        except requests.ConnectionError as e:
            raise OllamaError(
                f"Não consegui conectar ao Ollama em {self.url}.\n"
                "   Verifique se ele está instalado e rodando (comando: ollama serve)."
            ) from e

        with response:
            if response.status_code == 404:
                raise OllamaError(
                    f"O modelo '{self.model}' não está instalado no Ollama.\n"
                    f"   Baixe com: ollama pull {self.model}"
                )
            if response.status_code != 200:
                raise OllamaError(f"Ollama respondeu {response.status_code}: {response.text[:300]}")

            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise OllamaError(f"Erro do Ollama: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    break

    def generate(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))
