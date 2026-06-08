import os
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import load_vectorstore, answer_question

# load env vars
load_dotenv()

st.set_page_config(
    page_title="Research Paper Q&A (RAG)",
    layout="centered"
)

st.title("Research Paper Q&A System")
st.markdown(
    "Ask any question about the papers: **Transformers, BERT, GPT-3, RAG, "
    "Sentence-BERT, LoRA, Llama 2**"
)

st.divider()

# check api key
api_key = os.environ.get("GROQ_API_KEY", "")

if not api_key:
    st.warning(
        "**GROQ_API_KEY not found.**\n\n"
        "Please create a `.env` file in this folder and add:\n"
        "```\nGROQ_API_KEY='your_key_here'\n```"
    )

# check chroma db
if not os.path.exists("chroma_db"):
    st.error(
        "**Vector database not found.**\n\n"
        "Please run this first:\n"
        "```\npython ingest.py\n```"
    )
    st.stop()

# cache vectorstore
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

vectorstore = get_vectorstore()
st.success("Vector database loaded. Ready to answer questions!")

# example queries
st.markdown("### Example Questions")

example_queries = [
    "How does self-attention differ from recurrence?",
    "What problem does RAG solve?",
    "How does LoRA reduce training cost?",
    "What is the difference between BERT and GPT?",
    "What is a sentence embedding?",
    "How does Llama 2 handle safety?",
]

col1, col2 = st.columns(2)
for i, example in enumerate(example_queries):
    if i % 2 == 0:
        if col1.button(example, key=f"ex_{i}"):
            st.session_state["prefill_query"] = example
    else:
        if col2.button(example, key=f"ex_{i}"):
            st.session_state["prefill_query"] = example

st.divider()

# user input
default_query = st.session_state.get("prefill_query", "")

user_query = st.text_input(
    "Ask your question:",
    value=default_query,
    placeholder="e.g. How does self-attention work?"
)

ask_button = st.button("Ask", type="primary")

# generate answer
if ask_button and user_query.strip():
    if not api_key:
        st.error("Please set your GROQ_API_KEY in the .env file first.")
    else:
        with st.spinner("Searching papers and generating answer..."):
            answer, sources = answer_question(vectorstore, user_query)

        st.markdown("### Answer")
        st.write(answer)

        if sources:
            st.markdown("### Sources Used")
            for source in sources:
                st.markdown(f"- `{source}`")

elif ask_button and not user_query.strip():
    st.warning("Please enter a question.")

st.divider()
st.caption("Built with LangChain + ChromaDB + Sentence Transformers + Groq (Llama 3)")
