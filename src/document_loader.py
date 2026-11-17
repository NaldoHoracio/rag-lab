"""
@author: Edvonaldo Horacio (edvonaldohoracio@gmail.com)
@date: 16-11-2026
@description: Leitura de PDFs e divisão do texto em chunks. Cada chunk guarda
              o arquivo de origem e a página, para que a resposta final possa
              citar de onde veio a informação.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    source: str  # nome do arquivo PDF
    page: int  # página (começando em 1)
    chunk_id: str  # identificador único e estável: arquivo:página:índice

    def metadata(self) -> dict:
        """Metadados no formato aceito pelo ChromaDB (sem o texto)."""
        data = asdict(self)
        data.pop("text")
        return data


def clean_text(text: str) -> str:
    """Normaliza espaços e desfaz hifenização de quebra de linha."""
    text = re.sub(r"-\n(\w)", r"\1", text)  # "regis-\ntrador" -> "registrador"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Divide o texto em pedaços de até `chunk_size` caracteres com sobreposição.

    Tenta cortar em fronteiras naturais (parágrafo > linha > frase > espaço)
    para não partir palavras ou frases no meio.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap deve ser menor que chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            # procura o melhor ponto de corte na segunda metade da janela
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep, chunk_size // 2)
                if cut != -1:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Retorna uma lista de (número_da_página, texto) de um PDF."""
    import PyPDF2

    pages = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append((i, text))
    return pages


def chunk_pdf(pdf_path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    """Lê um PDF e devolve seus chunks, página por página."""
    chunks: list[Chunk] = []
    for page_num, text in load_pdf_pages(pdf_path):
        for j, piece in enumerate(split_text(text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    text=piece,
                    source=pdf_path.name,
                    page=page_num,
                    chunk_id=f"{pdf_path.name}:{page_num}:{j}",
                )
            )
    return chunks


def chunk_directory(data_path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    """Lê todos os PDFs de uma pasta e devolve a lista completa de chunks."""
    pdfs = sorted(Path(data_path).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {data_path}")
    all_chunks: list[Chunk] = []
    for i, pdf in enumerate(pdfs, start=1):
        pdf_chunks = chunk_pdf(pdf, chunk_size, overlap)
        print(f"{i}/{len(pdfs)}: {pdf.name} -> {len(pdf_chunks)} chunks")
        all_chunks.extend(pdf_chunks)
    return all_chunks
