# ============================================================
# AI COURSE INTELLIGENCE
# Free Cloud Multi-Document RAG Chatbot
# Developed by Ruturaj Mokashi
# ============================================================

import hashlib
import io
import logging
import uuid

import streamlit as st
import torch

from pypdf import PdfReader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Course Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. RAG SETTINGS
# Cloud adaptation of evaluated Configuration B
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVAL_K = 4

MAX_PDFS = 5
MAX_TOTAL_UPLOAD_MB = 50

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)

MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 250


# ============================================================
# 4. CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(150,150,160,0.15);
}

[data-testid="stMetric"] {
    border: 1px solid rgba(130,130,150,0.18);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    background: rgba(255,255,255,0.025);
}

.stButton > button {
    min-height: 48px;
    border-radius: 12px;
    font-weight: 600;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
}

button[data-baseweb="tab"] {
    font-weight: 600;
}

h1 {
    letter-spacing: -0.8px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 5. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "vectorstore": None,
    "document_count": 0,
    "page_count": 0,
    "chunk_count": 0,
    "last_upload_signature": None,
    "knowledge_base_ready": False,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    # Small CPU-friendly embedding model.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


# ============================================================
# 7. LOAD LOCAL LANGUAGE MODEL
# ============================================================

@st.cache_resource
def load_local_llm():

    # Load tokenizer from Hugging Face.
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
    )

    # Load the small Qwen model locally.
    #
    # torch_dtype="auto" preserves the model's compact
    # stored datatype instead of forcing float32.
    #
    # device_map="auto" uses CPU on Streamlit Cloud.
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype="auto",
        device_map="auto",
    )

    # Inference mode only.
    model.eval()

    return tokenizer, model


# ============================================================
# 8. TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)


# ============================================================
# 9. CREATE UPLOAD SIGNATURE
# ============================================================

def create_upload_signature(uploaded_files):

    if not uploaded_files:

        return None

    hasher = hashlib.sha256()

    # Sort filenames so ordering does not affect signature.
    for uploaded_file in sorted(
        uploaded_files,
        key=lambda file: file.name,
    ):

        file_bytes = uploaded_file.getvalue()

        hasher.update(
            uploaded_file.name.encode("utf-8")
        )

        hasher.update(file_bytes)

    return hasher.hexdigest()


# ============================================================
# 10. DELETE OLD VECTORSTORE
# ============================================================

def delete_existing_vectorstore():

    vectorstore = st.session_state.vectorstore

    if vectorstore is not None:

        try:

            vectorstore.delete_collection()

        except Exception:

            logger.exception(
                "Could not delete old Chroma collection."
            )

    st.session_state.vectorstore = None

    st.session_state.document_count = 0

    st.session_state.page_count = 0

    st.session_state.chunk_count = 0

    st.session_state.knowledge_base_ready = False

    st.session_state.messages = []


# ============================================================
# 11. LOAD PDF DOCUMENTS
# ============================================================

def load_pdf_documents(uploaded_files):

    documents = []

    readable_pages = 0

    for uploaded_file in uploaded_files:

        try:

            file_bytes = uploaded_file.getvalue()

            reader = PdfReader(
                io.BytesIO(file_bytes)
            )

            for page_number, page in enumerate(
                reader.pages
            ):

                # Extract text from one PDF page.
                text = page.extract_text()

                # Ignore pages without readable text.
                if not text:

                    continue

                if not text.strip():

                    continue

                document = Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_number,
                    },
                )

                documents.append(document)

                readable_pages += 1

        except Exception as error:

            logger.exception(
                "PDF processing failed for %s",
                uploaded_file.name,
            )

            raise ValueError(
                f"Could not read "
                f"'{uploaded_file.name}'."
            ) from error

    return documents, readable_pages


# ============================================================
# 12. BUILD VECTORSTORE
# ============================================================

def build_vectorstore(uploaded_files):

    # Load PDF pages.
    documents, page_count = (
        load_pdf_documents(uploaded_files)
    )

    if not documents:

        raise ValueError(
            "No readable text was found "
            "inside the uploaded PDFs."
        )

    # Split pages into Configuration-B chunks.
    splits = text_splitter.split_documents(
        documents
    )

    if not splits:

        raise ValueError(
            "No text chunks could be created."
        )

    # Load embeddings only when we actually build the KB.
    embeddings = load_embeddings()

    # Every build receives a new Chroma collection.
    collection_name = (
        "rag_" + uuid.uuid4().hex
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # Smaller batches reduce memory spikes.
    batch_size = 32

    total_chunks = len(splits)

    progress_bar = st.progress(
        0,
        text="Creating document embeddings...",
    )

    for start in range(
        0,
        total_chunks,
        batch_size,
    ):

        batch = splits[
            start:start + batch_size
        ]

        vectorstore.add_documents(
            batch
        )

        processed = min(
            start + batch_size,
            total_chunks,
        )

        progress = (
            processed / total_chunks
        )

        progress_bar.progress(
            progress,
            text=(
                f"Indexing "
                f"{processed:,} / "
                f"{total_chunks:,} chunks"
            ),
        )

    progress_bar.empty()

    return (
        vectorstore,
        page_count,
        total_chunks,
    )


# ============================================================
# 13. FORMAT RETRIEVED CONTEXT
# ============================================================

def format_docs(docs):

    chunks = []

    for doc in docs:

        # Remove unnecessary PDF whitespace.
        clean_text = " ".join(
            doc.page_content.split()
        )

        chunks.append(clean_text)

    return "\n\n".join(chunks)


# ============================================================
# 14. GET REAL SOURCE METADATA
# ============================================================

def get_retrieved_sources(docs):

    sources = []

    for doc in docs:

        filename = doc.metadata.get(
            "source",
            "Unknown document",
        )

        # PDF metadata uses zero-based page indexing.
        page = (
            doc.metadata.get(
                "page",
                0,
            )
            + 1
        )

        citation = (
            f"{filename} — Page {page}"
        )

        # Avoid duplicate source labels.
        if citation not in sources:

            sources.append(citation)

    return sources


# ============================================================
# 15. GENERATE ANSWER LOCALLY
# ============================================================

def generate_answer(
    question,
    context,
):

    # Model is loaded only when the first question is asked.
    tokenizer, model = load_local_llm()

    system_prompt = """
You are a document question-answering assistant.

Follow these rules strictly:

1. Answer ONLY using information in the supplied context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Give a clear and simple answer.
5. Prefer short paragraphs or bullet points.
6. Do NOT include filenames.
7. Do NOT include page numbers.
8. Do NOT create citations.

If the context does not clearly contain the answer,
respond exactly with:

"The answer is not available in the provided course materials."
"""

    user_prompt = f"""
CONTEXT:

{context}

QUESTION:

{question}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    # Qwen's own chat template formats system/user messages.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Convert prompt into model tokens.
    model_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    # Put tensors on the same device as the model.
    model_inputs = {
        key: value.to(model.device)
        for key, value in model_inputs.items()
    }

    input_length = (
        model_inputs["input_ids"].shape[-1]
    )

    # No gradients are required for inference.
    with torch.inference_mode():

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove the original prompt tokens.
    answer_tokens = generated_ids[
        0,
        input_length:
    ]

    # Convert generated tokens back into text.
    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True,
    ).strip()

    if not answer:

        raise RuntimeError(
            "The local model returned "
            "an empty response."
        )

    return answer


# ============================================================
# 16. SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🧠 AI Course Intelligence"
    )

    st.caption(
        "Developed by **Ruturaj Mokashi**"
    )

    st.divider()

    st.subheader("About")

    st.write(
        "Upload PDF documents and explore them "
        "using Retrieval-Augmented Generation."
    )

    st.divider()

    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    st.subheader(
        "📄 Knowledge Base"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Upload up to 5 PDFs. "
            "Maximum combined size: 50 MB."
        ),
    )

    # Maximum of 5 PDFs.
    too_many_files = (
        uploaded_files
        and len(uploaded_files) > MAX_PDFS
    )

    if too_many_files:

        st.error(
            f"Please upload no more than "
            f"{MAX_PDFS} PDFs."
        )

    # Calculate total size.
    total_upload_bytes = sum(
        len(file.getvalue())
        for file in uploaded_files
    )

    total_upload_mb = (
        total_upload_bytes
        / (1024 * 1024)
    )

    upload_too_large = (
        total_upload_mb
        > MAX_TOTAL_UPLOAD_MB
    )

    if upload_too_large:

        st.error(
            f"Combined upload size must be "
            f"below {MAX_TOTAL_UPLOAD_MB} MB."
        )

    if uploaded_files:

        st.caption(
            f"{len(uploaded_files)} PDF(s) • "
            f"{total_upload_mb:.1f} MB total"
        )

    # Detect whether uploaded files changed.
    current_signature = (
        create_upload_signature(
            uploaded_files
        )
    )

    if (
        current_signature
        != st.session_state.last_upload_signature
    ):

        delete_existing_vectorstore()

        st.session_state.last_upload_signature = (
            current_signature
        )

    build_disabled = (
        not uploaded_files
        or too_many_files
        or upload_too_large
    )

    if st.button(
        "⚙️ Build Knowledge Base",
        type="primary",
        use_container_width=True,
        disabled=build_disabled,
    ):

        with st.spinner(
            "Reading and preparing PDFs..."
        ):

            try:

                delete_existing_vectorstore()

                (
                    vectorstore,
                    page_count,
                    chunk_count,
                ) = build_vectorstore(
                    uploaded_files
                )

                st.session_state.vectorstore = (
                    vectorstore
                )

                st.session_state.document_count = (
                    len(uploaded_files)
                )

                st.session_state.page_count = (
                    page_count
                )

                st.session_state.chunk_count = (
                    chunk_count
                )

                st.session_state.knowledge_base_ready = (
                    True
                )

                st.success(
                    "Knowledge base ready."
                )

            except Exception:

                logger.exception(
                    "Knowledge-base creation failed."
                )

                st.error(
                    "The PDFs could not be processed. "
                    "Please check the files and try again."
                )

    st.divider()

    # --------------------------------------------------------
    # SESSION STATS
    # --------------------------------------------------------

    st.subheader(
        "Current Session"
    )

    st.write(
        f"📚 **"
        f"{st.session_state.document_count}"
        f" documents**"
    )

    st.write(
        f"📄 **"
        f"{st.session_state.page_count:,}"
        f" pages**"
    )

    st.write(
        f"🧩 **"
        f"{st.session_state.chunk_count:,}"
        f" chunks**"
    )

    st.write(
        f"🔎 **Top "
        f"{RETRIEVAL_K} retrieval**"
    )

    st.divider()

    # --------------------------------------------------------
    # AI STACK
    # --------------------------------------------------------

    st.subheader(
        "AI Stack"
    )

    st.caption("Language Model")

    st.code(
        "Qwen2.5-0.5B-Instruct"
    )

    st.caption("Embeddings")

    st.code(
        "all-MiniLM-L6-v2"
    )

    st.caption("Vector Database")

    st.code("Chroma")

    st.caption("Framework")

    st.code(
        "LangChain + Streamlit"
    )

    st.caption(
        "RAG Configuration"
    )

    st.code(
        "chunk=800 | overlap=100 | k=4"
    )

    st.caption(
        "☁️ 100% local inference — no paid API"
    )

    st.caption(
        "First answer may take longer "
        "while the language model loads."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 17. MAIN HEADER
# ============================================================

with st.container(
    border=True
):

    st.title(
        "🧠 AI Course Intelligence"
    )

    st.markdown(
        """
### Multi-Document RAG Assistant

Upload PDF documents and explore them using
**Retrieval-Augmented Generation**.

The assistant retrieves relevant document context first
and then generates a grounded answer locally.
"""
    )

    st.caption(
        "RAG • Multi-Document Search • "
        "LangChain • Transformers • "
        "Chroma • Streamlit"
    )


# ============================================================
# 18. METRICS
# ============================================================

metric1, metric2, metric3, metric4 = (
    st.columns(4)
)


with metric1:

    st.metric(
        "Documents",
        st.session_state.document_count,
    )


with metric2:

    st.metric(
        "Pages",
        f"{st.session_state.page_count:,}",
    )


with metric3:

    st.metric(
        "Knowledge Chunks",
        f"{st.session_state.chunk_count:,}",
    )


with metric4:

    st.metric(
        "Retrieved Context",
        f"Top {RETRIEVAL_K}",
    )


# ============================================================
# 19. KB STATUS
# ============================================================

if not st.session_state.knowledge_base_ready:

    st.info(
        "👈 Upload PDF documents in the sidebar "
        "and click **Build Knowledge Base**."
    )


# ============================================================
# 20. TOPIC EXPLORER
# ============================================================

st.subheader(
    "💡 Explore the Knowledge Base"
)

st.caption(
    "Switch between topics or ask "
    "your own question."
)


topic_tabs = st.tabs(
    [
        "🧠 Large Language Models",
        "✍️ Prompt Engineering",
        "🤖 AI Agents",
        "🔗 Hugging Face",
        "⚙️ Agentic Design",
        "📚 Applications",
    ]
)


selected_question = None


with topic_tabs[0]:

    st.markdown(
        """
### 🧠 Large Language Models

Explore how Large Language Models understand
and generate language.
"""
    )

    if st.button(
        "What is a Large Language Model?",
        key="llm_question",
        use_container_width=True,
    ):

        selected_question = (
            "What is a Large Language Model "
            "and how does it work?"
        )


with topic_tabs[1]:

    st.markdown(
        """
### ✍️ Prompt Engineering

Explore how prompts influence model behaviour
and response quality.
"""
    )

    if st.button(
        "What is Prompt Engineering?",
        key="prompt_question",
        use_container_width=True,
    ):

        selected_question = (
            "What is prompt engineering "
            "and why is it important?"
        )


with topic_tabs[2]:

    st.markdown(
        """
### 🤖 AI Agents

Explore intelligent systems that can reason,
use tools and perform tasks.
"""
    )

    if st.button(
        "What are AI Agents?",
        key="agents_question",
        use_container_width=True,
    ):

        selected_question = (
            "What are AI agents and "
            "how do agentic systems work?"
        )


with topic_tabs[3]:

    st.markdown(
        """
### 🔗 Hugging Face

Explore pretrained models, Transformers
and the Hugging Face ecosystem.
"""
    )

    if st.button(
        "What is Hugging Face?",
        key="hf_question",
        use_container_width=True,
    ):

        selected_question = (
            "What is Hugging Face "
            "and how is it used with LLMs?"
        )


with topic_tabs[4]:

    st.markdown(
        """
### ⚙️ Agentic Design Patterns

Explore reusable patterns for building
AI-agent systems.
"""
    )

    if st.button(
        "What are Agentic Design Patterns?",
        key="agentic_question",
        use_container_width=True,
    ):

        selected_question = (
            "What are the main "
            "agentic design patterns?"
        )


with topic_tabs[5]:

    st.markdown(
        """
### 📚 AI Applications

Explore practical AI applications described
inside your uploaded documents.
"""
    )

    if st.button(
        "Show common LLM applications",
        key="applications_question",
        use_container_width=True,
    ):

        selected_question = (
            "What are common real-world applications "
            "of Large Language Models?"
        )


# ============================================================
# 21. CHAT
# ============================================================

st.divider()

st.subheader(
    "💬 Ask the Document Assistant"
)

st.caption(
    "Answers are grounded in the uploaded PDF documents."
)


# Display previous messages.
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


typed_question = st.chat_input(
    "Ask a question about your uploaded documents...",
    disabled=(
        not st.session_state.knowledge_base_ready
    ),
)


question = (
    typed_question
    if typed_question
    else selected_question
)


# ============================================================
# 22. PROCESS QUESTION
# ============================================================

if question:

    if not st.session_state.knowledge_base_ready:

        st.warning(
            "Please build the knowledge base first."
        )

    else:

        # Show user question.
        with st.chat_message("user"):

            st.markdown(question)

        # Store user question.
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # Generate assistant response.
        with st.chat_message(
            "assistant"
        ):

            try:

                with st.spinner(
                    "Searching documents and "
                    "generating the answer..."
                ):

                    # Configuration B:
                    # retrieve the top 4 chunks.
                    retriever = (
                        st.session_state
                        .vectorstore
                        .as_retriever(
                            search_type="similarity",
                            search_kwargs={
                                "k": RETRIEVAL_K,
                            },
                        )
                    )

                    retrieved_docs = (
                        retriever.invoke(
                            question
                        )
                    )

                    context = format_docs(
                        retrieved_docs
                    )

                    answer = generate_answer(
                        question,
                        context,
                    )

                    retrieved_sources = (
                        get_retrieved_sources(
                            retrieved_docs
                        )
                    )

                # Display clean answer.
                st.markdown(answer)

                # Display sources from metadata.
                st.markdown(
                    "#### 📚 Retrieved Sources"
                )

                for source in retrieved_sources:

                    st.markdown(
                        f"- `{source}`"
                    )

                # Display raw retrieved chunks.
                with st.expander(
                    "🔎 View Retrieved Context"
                ):

                    for index, doc in enumerate(
                        retrieved_docs,
                        start=1,
                    ):

                        filename = (
                            doc.metadata.get(
                                "source",
                                "Unknown document",
                            )
                        )

                        page = (
                            doc.metadata.get(
                                "page",
                                0,
                            )
                            + 1
                        )

                        preview = " ".join(
                            doc.page_content.split()
                        )

                        st.markdown(
                            f"**{index}. "
                            f"{filename} — "
                            f"Page {page}**"
                        )

                        st.caption(
                            preview[:600] + "..."
                        )

                        st.divider()

                # Save sources with assistant response.
                source_text = "\n".join(
                    f"- `{source}`"
                    for source in retrieved_sources
                )

                saved_answer = (
                    answer
                    + "\n\n"
                    + "#### 📚 Retrieved Sources\n"
                    + source_text
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": saved_answer,
                    }
                )

            except Exception:

                logger.exception(
                    "Question-answering request failed."
                )

                st.error(
                    "The local AI model could not complete "
                    "this request. Check the application "
                    "logs for details."
                )


# ============================================================
# 23. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Course Intelligence • "
    "Multi-Document RAG System"
)

st.caption(
    "Developed by Ruturaj Mokashi"
)

st.caption(
    "LangChain • Transformers • "
    "Chroma • Streamlit • "
    "Source-Aware Retrieval"
)