import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from document_loader import Chunk
from generator import OllamaError, OllamaGenerator, build_messages, format_context

HITS = [
    (Chunk("The ESP32 has 18 ADC channels.", "esp32.pdf", 12, "esp32.pdf:12:0"), 0.81),
    (Chunk("Timer1 CTC mode uses OCR1A.", "atmega.pdf", 95, "atmega.pdf:95:0"), 0.40),
]


def test_contexto_numerado_com_arquivo_e_pagina():
    ctx = format_context(HITS)
    assert "[1] (arquivo: esp32.pdf, página 12)" in ctx
    assert "[2] (arquivo: atmega.pdf, página 95)" in ctx


def test_mensagens_tem_system_e_pergunta():
    messages = build_messages("Quantos ADCs?", HITS)
    assert messages[0]["role"] == "system" and "português" in messages[0]["content"]
    assert messages[1]["content"].endswith("Pergunta: Quantos ADCs?")


class FakeOllama(BaseHTTPRequestHandler):
    """Imita o endpoint /api/chat do Ollama, respondendo em streaming (NDJSON)."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.last_request = body
        if body["model"] != "modelo-ok":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "model not found"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for piece in ["O ESP32 tem ", "18 canais ", "de ADC [1]."]:
            self.wfile.write(json.dumps({"message": {"content": piece}, "done": False}).encode() + b"\n")
        self.wfile.write(json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n")

    def log_message(self, *args):
        pass


@pytest.fixture
def ollama_url():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_streaming_junta_os_pedacos(ollama_url):
    gen = OllamaGenerator(ollama_url, "modelo-ok", temperature=0.2)
    assert gen.generate(build_messages("Quantos ADCs?", HITS)) == "O ESP32 tem 18 canais de ADC [1]."
    assert FakeOllama.last_request["options"]["temperature"] == 0.2
    assert FakeOllama.last_request["stream"] is True


def test_modelo_inexistente_sugere_pull(ollama_url):
    gen = OllamaGenerator(ollama_url, "modelo-inexistente")
    with pytest.raises(OllamaError, match="ollama pull modelo-inexistente"):
        gen.generate(build_messages("x", HITS))


def test_ollama_desligado():
    gen = OllamaGenerator("http://127.0.0.1:9", "modelo-ok")  # porta sem nada escutando
    with pytest.raises(OllamaError, match="ollama serve"):
        gen.generate(build_messages("x", HITS))
