
import uuid
import os
import streamlit as st
from dotenv import load_dotenv

# Load .env file (must be before any os.getenv() calls)
load_dotenv()

# ── API Key check — fail early with a helpful message ─────────────────
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    st.error(
        "❌ **HUGGINGFACEHUB_API_TOKEN not found.**\n\n"
        "1. Copy `.env.example` to `.env`\n"
        "2. Add your HuggingFace token from https://huggingface.co/settings/tokens\n"
        "3. Restart the app"
    )
    st.stop()

# Import after the key check so we don't crash on missing key
from rag_chain import create_rag_chain, get_response, AVAILABLE_MODELS
from ingest import ingest_documents, ingest_from_folder

# ── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Customer Support Chatbot")
st.caption("Powered by **RAG** · **Gemini** · **ChromaDB** · **LangChain**")

# ── Session state initialization ──────────────────────────────────────
# Streamlit reruns the script on every interaction.
# session_state persists variables across reruns.

# Chat history — list of {"role": "user"/"assistant", "content": str, "sources": list}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Unique session ID — used to keep each user's memory separate
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Selected question-answering model (changeable from the sidebar)
if "model" not in st.session_state:
    st.session_state.model = AVAILABLE_MODELS[0]

# The RAG chain — built once per selected model, reused across all messages
if "chain" not in st.session_state:
    with st.spinner("🔧 Initializing RAG pipeline... (first load takes a moment)"):
        st.session_state.chain = create_rag_chain(st.session_state.model)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Knowledge Base")
    st.caption(
        "Upload your support documents (PDFs, FAQs, manuals) "
        "to teach the chatbot about your products and policies."
    )

    # ── File upload ───────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload multiple files at once. They'll be chunked and stored in ChromaDB."
    )

    if uploaded_files:
        if st.button("📥 Process Uploaded Files", use_container_width=True):
            with st.spinner(f"Embedding {len(uploaded_files)} file(s)..."):
                count = ingest_documents(uploaded_files)
            st.success(f"✅ {count} chunks stored in ChromaDB!")

    st.divider()

    # ── Load from local folder ────────────────────────────────────────
    st.caption("Or pre-load documents from your local `knowledge_base/` folder:")
    if st.button("📂 Load from /knowledge_base", use_container_width=True):
        with st.spinner("Loading documents from folder..."):
            count = ingest_from_folder("./knowledge_base")
        if count:
            st.success(f"✅ {count} chunks loaded!")
        else:
            st.info("No .pdf or .txt files found in /knowledge_base folder.")

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")

    # Model selector — switching rebuilds the RAG chain with the new model
    selected_model = st.selectbox(
        "Answering model (HuggingFace)",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.model),
        help="Pick which free HuggingFace model answers your questions."
    )
    if selected_model != st.session_state.model:
        st.session_state.model = selected_model
        with st.spinner(f"Switching to {selected_model}..."):
            st.session_state.chain = create_rag_chain(selected_model)
        st.rerun()

    show_sources = st.toggle(
        "Show source documents",
        value=True,
        help="Show which documents were used to generate each answer"
    )

    st.divider()

    # ── Clear chat ────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        # Generate new session ID to reset conversation memory in the chain too
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    # ── Info box ──────────────────────────────────────────────────────
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Add documents** — Upload PDFs/TXTs or put them in `knowledge_base/`
        2. **Process them** — Click the process/load button
        3. **Ask questions** — Type in the chat box below
        
        The chatbot will answer using ONLY your uploaded documents.
        It will say it doesn't know if the answer isn't in the docs.
        """)

    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")

# ── Chat message display ───────────────────────────────────────────────
# Render all previous messages in the conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # Renders as user/assistant bubble
        st.write(msg["content"])

        # Show source documents under assistant messages (if enabled)
        if show_sources and msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📄 {len(msg['sources'])} source(s) used"):
                for src in msg["sources"]:
                    st.caption(f"• {src}")

# ── Chat input & response ──────────────────────────────────────────────
# st.chat_input() returns the text when user hits Enter, else None
if prompt := st.chat_input("Ask a question about our products or policies..."):

    # 1. Immediately display the user's message
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": []
    })

    # 3. Get response from RAG chain and display it
    with st.chat_message("assistant"):
        try:
            with st.spinner("🔍 Searching knowledge base..."):
                result = get_response(
                    st.session_state.chain,
                    prompt,
                    st.session_state.session_id
                )
        except Exception as e:
            # Don't crash the whole app (which would wipe the chat on screen).
            # Show a friendly message and keep the conversation intact.
            msg = str(e)
            if any(s in msg for s in ("503", "UNAVAILABLE", "high demand")):
                friendly = ("⚠️ The Gemini model is temporarily overloaded (503). "
                            "Please ask again in a few seconds.")
            elif any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "quota")):
                friendly = ("⚠️ Daily free-tier quota reached for this model. "
                            "Try again later, or set a different `GEMINI_MODEL` in `.env` "
                            "and restart.")
            else:
                friendly = f"⚠️ Something went wrong: {msg[:300]}"
            st.warning(friendly)
            result = {"answer": friendly, "sources": []}

        st.write(result["answer"])

        if show_sources and result.get("sources"):
            with st.expander(f"📄 {len(result['sources'])} source(s) used"):
                for src in result["sources"]:
                    st.caption(f"• {src}")

    # 4. Save assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", [])
    })
