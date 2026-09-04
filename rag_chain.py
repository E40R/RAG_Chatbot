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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from ingest import ResilientEmbeddings
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


def create_rag_chain():
    """
    Build and return the full conversational RAG chain.
    Call this ONCE at startup and reuse the returned chain.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Check your .env file.")

    # ── 1. LLM — Gemini 1.5 Flash ────────────────────────────────────
    # temperature=0 → fully deterministic, factual answers (good for support)
    # temperature=1 → more creative/varied (good for creative tasks)
    # We use 0.2 — mostly factual with slight natural variation
    # NOTE on model choice: the free tier caps each model at ~20 requests/day,
    # and different API keys expose different models (some keys 404 on
    # gemini-2.5-flash-lite but serve gemini-flash-latest). So the model is
    # configurable via the GEMINI_MODEL env var — if one model is exhausted or
    # unavailable on your key, set GEMINI_MODEL to another (e.g. gemini-flash-latest,
    # gemini-flash-lite-latest, gemini-2.5-flash) and restart; no code change needed.
    # max_retries lets the client ride out short rate-limit (429) bursts.
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        google_api_key=api_key,
        temperature=0.2,
        max_retries=3,
    )

    # ── 2. Embedding model ────────────────────────────────────────────
    # MUST be the same model used during ingestion!
    # If you ingest with model A but query with model B → wrong vectors → garbage results
    embeddings = ResilientEmbeddings(
        model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
        google_api_key=api_key
    )

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
