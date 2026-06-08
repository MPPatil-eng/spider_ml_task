import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# config settings
PAPERS_FOLDER = "papers"
CHROMA_DB_PATH = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_all_pdfs(folder_path):
    """Load all PDFs."""
    all_docs = []

    if not os.path.exists(folder_path):
        print(f"ERROR: Folder '{folder_path}' not found. Please create it and add your PDFs.")
        return all_docs

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"ERROR: No PDF files found in '{folder_path}'. Please add your papers.")
        return all_docs

    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        print(f"  Loading: {pdf_file}")

        loader = PyPDFLoader(full_path)
        docs = loader.load()

        # tag source file
        for doc in docs:
            doc.metadata["source"] = pdf_file

        all_docs.extend(docs)

    print(f"\nTotal pages loaded: {len(all_docs)}")
    return all_docs


def split_into_chunks(docs):
    """Split into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def store_in_chromadb(chunks):
    """Embed and store."""
    print("\nLoading embedding model (this may take a moment the first time)...")

    # local embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Creating vector database and storing chunks...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"\nDone! Vector database saved to '{CHROMA_DB_PATH}/'")
    return vectorstore


if __name__ == "__main__":
    # step 1: load
    print("=" * 50)
    print("STEP 1: Loading PDFs...")
    print("=" * 50)
    docs = load_all_pdfs(PAPERS_FOLDER)

    if not docs:
        print("No documents loaded. Exiting.")
        exit()

    # step 2: split
    print("\n" + "=" * 50)
    print("STEP 2: Splitting into chunks...")
    print("=" * 50)
    chunks = split_into_chunks(docs)

    # step 3: store
    print("\n" + "=" * 50)
    print("STEP 3: Storing in ChromaDB...")
    print("=" * 50)
    store_in_chromadb(chunks)

    print("\nIngestion complete! You can now run: streamlit run app.py")
