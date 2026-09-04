"""
ingest.py — Document Loading, Chunking, Embedding, and Storage
===============================================================
Two ways to use this:
  1. Called by app.py (Streamlit) when user uploads files via UI
  2. Run directly: python ingest.py  (to pre-load your knowledge_base/ folder)

What this does:
  PDF/TXT files
      → split into overlapping chunks (so context isn't lost at boundaries)
      → each chunk converted to an embedding vector (via Gemini)
      → stored in ChromaDB (persisted to ./chroma_db folder)
"""
import os
import time
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ── Config ────────────────────────────────────────────────────────────
CHROMA_DIR    = "./chroma_db"   # Where ChromaDB saves its data
CHUNK_SIZE    = 1000            # Max characters per chunk
CHUNK_OVERLAP = 200             # Overlap so context isn't cut off between chunks


class ResilientEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Gemini embeddings with retry/backoff for free-tier rate limits.

    The Gemini embedding endpoint allows only ~5 requests/minute on the free
    tier and returns a 500/429 when that burst limit is hit. A short
    exponential backoff lets the per-minute window clear so the call succeeds
    instead of crashing the whole chain. Used by both ingestion and querying.
    """
    def _with_retry(self, fn, *args, **kwargs):
        delay = 2.0
        for attempt in range(5):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                transient = any(s in msg for s in
                                ("429", "500", "RESOURCE_EXHAUSTED", "Internal error"))
                if transient and attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        return None

    def embed_query(self, text, *args, **kwargs):
        return self._with_retry(super().embed_query, text, *args, **kwargs)

    def embed_documents(self, texts, *args, **kwargs):
        return self._with_retry(super().embed_documents, texts, *args, **kwargs)


def _get_embeddings():
    """Returns Gemini embedding model (converts text → vectors)."""
    return ResilientEmbeddings(
        model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def _get_vectorstore():
    """Returns ChromaDB connection (creates or loads existing DB)."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings()
    )


def _get_splitter():
    """
    Returns a text splitter.
    
    RecursiveCharacterTextSplitter tries to split on these characters IN ORDER:
      1. Double newline (paragraph breaks) — most preferred
      2. Single newline
      3. Period + space (sentence end)
      4. Space (word boundary)
      5. Individual characters — last resort
    
    This means it tries to keep natural language units intact.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def ingest_documents(uploaded_files) -> int:
    """
    Process Streamlit-uploaded file objects into ChromaDB.
    Called by app.py when user uploads files via the UI.

    Args:
        uploaded_files: list of Streamlit UploadedFile objects

    Returns:
        Total number of chunks stored in ChromaDB
    """
    vectorstore = _get_vectorstore()
    splitter    = _get_splitter()
    total       = 0

    for file in uploaded_files:
        suffix = Path(file.name).suffix.lower()

        # Document loaders need a real file path (not a bytes buffer),
        # so we save to a temp file first, then delete it after.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                print(f"⚠️  Skipping unsupported type: {file.name}")
                continue

            docs = loader.load()

            # Tag each document with the original filename
            # This appears in the UI as "Sources used"
            for doc in docs:
                doc.metadata["source"] = file.name

            chunks = splitter.split_documents(docs)
            vectorstore.add_documents(chunks)
            total += len(chunks)
            print(f"✅ {file.name} → {len(chunks)} chunks")

        except Exception as e:
            print(f"❌ Failed to process {file.name}: {e}")

        finally:
            os.unlink(tmp_path)  # Always clean up temp file, even if error

    return total


def ingest_from_folder(folder_path: str) -> int:
    """
    Load all PDFs and TXTs from a local folder into ChromaDB.
    Called by app.py when user clicks "Load from knowledge_base folder".
    Also the entry point when run as a script.

    Args:
        folder_path: path to folder containing your documents

    Returns:
        Total number of chunks stored in ChromaDB
    """
    folder = Path(folder_path)

    # Create folder if it doesn't exist
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f"📁 Created folder: {folder_path}")
        print("   Add your PDF/TXT files there, then run again.")
        return 0

    vectorstore = _get_vectorstore()
    splitter    = _get_splitter()
    all_docs    = []

    # Load all PDFs
    for pdf_file in sorted(folder.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_file))
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_file.name
        all_docs.extend(docs)
        print(f"📄 Loaded PDF: {pdf_file.name} ({len(docs)} pages)")

    # Load all TXTs
    for txt_file in sorted(folder.glob("*.txt")):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source"] = txt_file.name
        all_docs.extend(docs)
        print(f"📄 Loaded TXT: {txt_file.name}")

    if not all_docs:
        print("⚠️  No .pdf or .txt files found in folder.")
        return 0

    chunks = splitter.split_documents(all_docs)
    vectorstore.add_documents(chunks)
    print(f"\n✅ Done! {len(chunks)} chunks stored in ChromaDB ({CHROMA_DIR})")
    return len(chunks)


# ── Run as standalone script ──────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 50)
    print("  RAG Chatbot — Document Ingestion")
    print("=" * 50)

    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found. Create a .env file first.")
        exit(1)

    ingest_from_folder("./knowledge_base")
