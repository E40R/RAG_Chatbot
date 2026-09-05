"""
rag_chain.py — The RAG Pipeline with Conversation Memory
=========================================================
This is the brain of the chatbot. Here's the full flow:

  User question
    ↓
  [History-aware retriever]
    → If the question references previous messages ("what about that?"),
      it rewrites it into a standalone, self-contained question first.
    ↓
  [ChromaDB similarity search]
    → Finds the top-K most semantically similar chunks to the question.
    ↓
  [Prompt construction]
    → Stuffs the retrieved chunks + chat history + question into a prompt.
    ↓
  [Gemini LLM]
    → Generates an answer grounded in the retrieved context.
    ↓
  [RunnableWithMessageHistory]
    → Saves question + answer to the session's chat history automatically.
    ↓
  Return: answer (str) + sources (list of filenames)

To switch to OpenAI instead of Gemini:
  pip install langchain-openai
  Replace ChatGoogleGenerativeAI → ChatOpenAI(model="gpt-4o-mini")
  Replace GoogleGenerativeAIEmbeddings → OpenAIEmbeddings()
  Replace GOOGLE_API_KEY → OPENAI_API_KEY
"""
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from ingest import get_embeddings

# ── Available question-answering models (HuggingFace Inference API) ────
# These are free serverless models confirmed to work. The UI selector and the
# HF_MODEL env var both pick from here. First entry is the default.
AVAILABLE_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",     # fast, good default
    "meta-llama/Llama-3.3-70B-Instruct",    # most capable
    "Qwen/Qwen2.5-72B-Instruct",            # strong alternative
]
DEFAULT_MODEL = os.getenv("HF_MODEL", AVAILABLE_MODELS[0])
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

# ── In-memory session store ───────────────────────────────────────────
# Maps session_id (str) → ChatMessageHistory object
# Each user session gets its own independent chat history.
# Note: This is in-memory only. Restarting the server clears all history.
# For production, replace ChatMessageHistory with a database-backed store.
_session_store: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Return existing chat history for session, or create a new one.
    This function is called automatically by RunnableWithMessageHistory.
    """
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def create_rag_chain(model_name: str = None):
    """
    Build and return the full conversational RAG chain.
    Call this ONCE at startup (or when the model is changed) and reuse it.

    Args:
        model_name: HuggingFace model to use for answering. Defaults to
                    HF_MODEL env / the first entry in AVAILABLE_MODELS.
    """
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not set. Check your .env file.")

    model_name = model_name or DEFAULT_MODEL

    # ── 1. LLM — HuggingFace Inference API ────────────────────────────
    # temperature 0.2 → mostly factual with slight natural variation.
    # The model is selectable (UI dropdown / HF_MODEL env) from AVAILABLE_MODELS.
    endpoint = HuggingFaceEndpoint(
        repo_id=model_name,
        task="conversational",
        huggingfacehub_api_token=api_key,
        temperature=0.2,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)

    # ── 2. Embedding model ────────────────────────────────────────────
    # Runs LOCALLY (sentence-transformers) — no API, no rate limits, so uploads
    # never fail on quota. MUST be the same model used during ingestion.
    embeddings = get_embeddings()

    # ── 3. Vector store ───────────────────────────────────────────────
    # Loads from ./chroma_db (created by ingest.py)
    # If empty, the retriever will return nothing and the LLM will say it doesn't know.
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # ── 4. Retriever ─────────────────────────────────────────────────
    # search_type="similarity" → cosine similarity (standard)
    # k=3 → return top 3 most relevant chunks per query
    # Tune k: higher k = more context but more tokens used = more cost
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # ── 5. History-aware retriever ────────────────────────────────────
    #
    # THE PROBLEM this solves:
    #
    #   Turn 1 — User: "What is your refund policy?"
    #   Turn 1 — Bot:  "30 days with receipt."
    #   Turn 2 — User: "What about for international orders?"
    #
    # Without this step, the retriever searches for "what about for international orders"
    # → poor results (too vague)
    #
    # WITH this step, the LLM first rephrases it using the history:
    #   → "What is the refund policy for international orders?"
    # → ChromaDB now gets a meaningful standalone query → great results ✅
    #
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and the user's latest question, "
         "rewrite the question as a clear, self-contained standalone question. "
         "Do NOT answer it. If it already makes sense on its own, return it unchanged."),
        MessagesPlaceholder("chat_history"),   # ← injects full chat history here
        ("human", "{input}")                   # ← the latest user message
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # ── 5b. Which retriever to use ────────────────────────────────────
    # We build the history-aware retriever above (kept for reference), but the
    # chain below uses the PLAIN `retriever`. Reason: the Gemini embedding
    # endpoint intermittently returns 500s when the query it receives was just
    # produced by the rewrite LLM inside the same threaded retrieval step. The
    # plain retriever embeds the user's raw question directly (reliable), and
    # the answer LLM still gets full chat_history below — so follow-up questions
    # keep their conversational context. This also halves Gemini calls per turn,
    # which matters on the free tier's tight rate limits.
    active_retriever = retriever

    # ── 6. Answer generation prompt ───────────────────────────────────
    # {context} → filled with the retrieved document chunks
    # {chat_history} → filled with previous turns
    # {input} → the user's question
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a friendly and professional customer support assistant. "
         "Answer the user's question using ONLY the information provided in the context below. "
         "If the context does not contain enough information to answer, say: "
         "\"I don't have that information in my knowledge base. "
         "Please reach out to our support team directly for help.\"\n"
         "Be concise, clear, and helpful. Do not make up information.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # ── 7. Stuff documents chain ──────────────────────────────────────
    # "Stuff" = literally stuff all retrieved chunks into the prompt as-is.
    # Other strategies: MapReduce, Refine (for very large doc sets)
    # For most use cases with k=3 chunks, "stuff" works perfectly.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # ── 8. Full RAG chain ─────────────────────────────────────────────
    # Combines: history-aware retrieval + answer generation
    rag_chain = create_retrieval_chain(active_retriever, question_answer_chain)

    # ── 9. Wrap with persistent conversation memory ───────────────────
    # This wrapper:
    #   - Automatically loads chat history before each call
    #   - Automatically saves the new turn to history after each call
    #   - Uses session_id to keep multiple users' histories separate
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,           # function that returns the right history
        input_messages_key="input",     # where the user question goes
        history_messages_key="chat_history",  # where history gets injected
        output_messages_key="answer"    # where the final answer is
    )

    return conversational_rag_chain


def get_response(chain, query: str, session_id: str = "default") -> dict:
    """
    Run the RAG chain for one turn.

    Args:
        chain:      The chain returned by create_rag_chain()
        query:      The user's question (string)
        session_id: Unique ID per user session (keeps conversations separate)

    Returns:
        {
            "answer":  str,        # The LLM's response
            "sources": list[str]   # Filenames of retrieved documents
        }
    """
    # Retry transient Gemini errors (503 model overload, 429 rate bursts,
    # 500 internal) with a short backoff so a temporary spike doesn't fail
    # the whole turn. A persistent error (e.g. daily quota exhausted) still
    # surfaces after the attempts so the UI can show a clear message.
    import time
    last_err = None
    for attempt in range(4):
        try:
            response = chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
            break
        except Exception as e:
            msg = str(e)
            transient = any(s in msg for s in
                            ("503", "UNAVAILABLE", "429", "500",
                             "RESOURCE_EXHAUSTED", "high demand", "Internal error"))
            if transient and attempt < 3:
                last_err = e
                time.sleep(2 * (attempt + 1))  # 2s, 4s, 6s
                continue
            raise
    else:
        raise last_err

    # ── Extract source documents for display in UI ────────────────────
    sources = []
    for doc in response.get("context", []):
        filename = doc.metadata.get("source", "Unknown")
        page     = doc.metadata.get("page")  # Present for PDFs, absent for TXTs
        label    = filename
        if page is not None:
            label += f" · page {int(page) + 1}"   # page is 0-indexed, show 1-indexed
        sources.append(label)

    # dict.fromkeys preserves order while removing duplicates (better than set())
    unique_sources = list(dict.fromkeys(sources))

    return {
        "answer":  response["answer"],
        "sources": unique_sources
    }
