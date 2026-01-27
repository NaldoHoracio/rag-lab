"""
@author: Edvonaldo Horacio (edvonaldohoracio@gmail.com)
@date: 27-10-2026
@description: Este modulo contem funcoes para preparar documentos para realizar o RAG
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def ingest_documents(data_path="./data/data-raw", persist_directory="./chroma_db"):
    # 1. Carregar documentos (Ex: Manuais do Mark II em PDF)
    print("📦 Carregando documentos...")
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # 2. Dividir em Chunks
    # Usamos o RecursiveCharacterTextSplitter para manter parágrafos unidos
    print("✂️ Criando chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Inicializar Embeddings (Usando a chave da OpenAI que você configurou)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. Criar e Persistir o ChromaDB
    print(f"💾 Salvando no ChromaDB em: {persist_directory}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("✅ Ingestão concluída com sucesso!")
    return vector_store

if __name__ == "__main__":
    ingest_documents()
