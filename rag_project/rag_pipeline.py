import os
from groq import Groq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# load env vars
load_dotenv()

# config settings
CHROMA_DB_PATH = "chroma_db"
TOP_K_RESULTS = 5


def load_vectorstore():
    """Load ChromaDB store."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model
    )

    return vectorstore


def retrieve_relevant_chunks(vectorstore, query):
    """Similarity search."""
    return vectorstore.similarity_search(query, k=TOP_K_RESULTS)


def build_prompt(query, retrieved_chunks):
    """Build LLM prompt."""
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.metadata.get("source", "Unknown Paper")
        context_text += f"\n--- Chunk {i+1} (from: {source}) ---\n"
        context_text += chunk.page_content
        context_text += "\n"

    prompt = f"""You are a helpful AI assistant that answers questions about ML research papers.

You are given some excerpts from research papers below. Use ONLY these excerpts to answer the question.
If the answer is not found in the context, say: "I couldn't find enough information in the provided papers."

Always mention which paper(s) you used at the end of your answer.

---
CONTEXT FROM PAPERS:
{context_text}
---

USER QUESTION: {query}

YOUR ANSWER:"""

    return prompt


def get_answer_from_groq(prompt):
    """Call Groq API."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions about ML research papers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # first completion
    return response.choices[0].message.content


def answer_question(vectorstore, query):
    """Full RAG pipeline."""
    chunks = retrieve_relevant_chunks(vectorstore, query)

    if not chunks:
        return "No relevant information found in the papers.", []

    prompt = build_prompt(query, chunks)
    answer = get_answer_from_groq(prompt)

    # collect sources
    sources = list(set(
        chunk.metadata.get("source", "Unknown")
        for chunk in chunks
    ))

    return answer, sources
