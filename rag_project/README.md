# Research Paper Q&A (RAG)

A Streamlit app that answers questions about ML research papers using Retrieval-Augmented Generation (RAG).

Papers included: Transformers, BERT, GPT-3, RAG, Sentence-BERT, LoRA, and Llama 2.

## Setup

1. Clone the repo and enter the project folder:

```bash
git clone <your-repo-url>
cd rag_project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` (get one free at [console.groq.com](https://console.groq.com)).

4. Build the vector database (run once):

```bash
python ingest.py
```

5. Start the app:

```bash
streamlit run app.py
```

## Project structure

```
rag_project/
├── app.py            # Streamlit UI
├── ingest.py         # PDF loading and vector DB creation
├── rag_pipeline.py   # Retrieval + Groq generation
├── papers/           # Research paper PDFs
├── requirements.txt
└── .env              # API key (not committed)
```

## Stack

- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Vector DB:** ChromaDB
- **LLM:** Groq `llama-3.1-8b-instant`
- **UI:** Streamlit
