# RAG

Repositório dedicado ao estudo e implementação de técnicas de **RAG (Retrieval-Augmented Generation)**. O objetivo é documentar o progresso desde conceitos fundamentais de indexação até arquiteturas avançadas de recuperação de dados para LLMs.

## Estrutura do Projeto

rag-lab/
├── chroma_db/               # Banco vetorial gerado pela indexação (não versionado)
├── data/
│   └── data-raw/            # PDFs a indexar (não versionados)
├── src/
│   ├── config.py            # Configurações centrais (caminhos, chunking, modelos), lidas do .env
│   ├── document_loader.py   # Leitura dos PDFs e divisão em chunks com arquivo e página
│   ├── embeddings.py        # Geração de embeddings (HuggingFace local ou OpenAI)
│   ├── vector_store.py      # Banco vetorial ChromaDB persistente
│   ├── build_index.py       # Script de indexação: PDFs → chunks → embeddings → ChromaDB
│   ├── generator.py         # Montagem do prompt e geração da resposta via Ollama
│   ├── ask.py               # Script de perguntas pelo terminal (retrieval + geração)
│   ├── ingestion.py         # Versão inicial da ingestão com LangChain
│   ├── prep_docs.py         # Versão inicial da preparação de documentos com OpenAI
│   └── main.py              # Versão inicial do fluxo de RAG com OpenAI
├── tests/
│   ├── conftest.py          # Adiciona src/ ao caminho de importação dos testes
│   ├── test_rag_lab.py      # Testes de chunking, indexação e banco vetorial
│   └── test_generator.py    # Testes da geração (com um servidor Ollama simulado)
├── .env                     # Variáveis de ambiente locais (não versionado)
├── requirements.txt
└── README.md

## Tecnologias Utilizadas

Linguagem: Python 3.12+

Frameworks de LLM: LangChain / LlamaIndex

Modelos: OpenAI (GPT-4o), Anthropic (Claude), Ollama (Llama 3.1)

Embeddings: HuggingFace, OpenAI, Nomic

Bancos Vetoriais: ChromaDB / FAISS

Avaliação: Ragas (Framework para métricas de Fidelidade e Relevância)