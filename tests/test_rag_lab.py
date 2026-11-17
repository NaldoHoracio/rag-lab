import pytest

from document_loader import Chunk, clean_text, split_text
from vector_store import ChromaStore


def test_split_respeita_tamanho_e_sobreposicao():
    text = " ".join(f"palavra{i}" for i in range(600))
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # o início de cada chunk deve aparecer no final do anterior (sobreposição)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[0] in prev


def test_split_texto_curto_e_vazio():
    assert split_text("curto", 100, 10) == ["curto"]
    assert split_text("   ", 100, 10) == []


def test_split_overlap_invalido():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_clean_text_desfaz_hifenizacao():
    assert clean_text("regis-\ntrador   de  controle") == "registrador de controle"


def test_chroma_add_e_search(tmp_path):
    store = ChromaStore(tmp_path / "db", "teste")
    chunks = [
        Chunk("GPIO max current", "esp32.pdf", 10, "esp32.pdf:10:0"),
        Chunk("Timer1 CTC mode", "atmega.pdf", 95, "atmega.pdf:95:0"),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])  # reindexar não duplica
    assert store.count() == 2

    hit, score = store.search([0.9, 0.1], k=1)[0]
    assert hit.source == "esp32.pdf" and hit.page == 10
    assert score > 0.9


class FakeEmbedder:
    """Embedder falso: conta palavras-chave, sem chamar API."""

    name = "fake/keywords"

    def encode(self, texts):
        return [[t.count("GPIO") + 0.01, t.count("Timer1") + 0.01] for t in texts]


@pytest.fixture
def pdfs(tmp_path):
    fpdf = pytest.importorskip("fpdf")
    path = tmp_path / "demo.pdf"
    pdf = fpdf.FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in ["The ESP32 GPIO pins source 40 mA. " * 40, "Timer1 CTC mode uses OCR1A. " * 40]:
        pdf.add_page()
        pdf.multi_cell(0, 6, text)
    pdf.output(str(path))
    return [path]


def test_build_index_e_skip_existing(tmp_path, pdfs):
    from build_index import build_index

    store = ChromaStore(tmp_path / "db", "teste", FakeEmbedder.name)
    summary = build_index(pdfs, store, FakeEmbedder(), chunk_size=500, overlap=50)
    assert summary["demo.pdf"] == store.count() > 2

    # segunda execução com skip_existing não reprocessa
    assert build_index(pdfs, store, FakeEmbedder(), skip_existing=True) == {}

    hit, _ = store.search(FakeEmbedder().encode(["Timer1"])[0], k=1)[0]
    assert hit.page == 2


def test_dry_run_nao_precisa_de_store(pdfs):
    from build_index import build_index

    summary = build_index(pdfs, dry_run=True, chunk_size=500, overlap=50)
    assert summary["demo.pdf"] > 2


def test_modelo_diferente_gera_erro(tmp_path):
    ChromaStore(tmp_path / "db", "teste", "openai/text-embedding-3-small")
    with pytest.raises(ValueError, match="--reset"):
        ChromaStore(tmp_path / "db", "teste", "huggingface/outro-modelo")
