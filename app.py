# ============================================================
# DAY 6 PROJECT 3
# AI COURSE INTELLIGENCE
# Multi-Document RAG Chatbot
# Developed by Ruturaj Mokashi
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import streamlit as st
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Know Your Artificial Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. PROJECT SETTINGS
# ============================================================

DOCUMENT_COUNT = 5
PAGE_COUNT = 1031

PERSIST_DIRECTORY = (
    r"C:\Ruturaj\Projects\AI Applications"
    r"\Dozenten_Uploads-20261006"
    r"\KI__Woche_2__Lerntag_6"
    r"\chroma_day6"
)

COLLECTION_NAME = "day6_course_chatbot"


# ============================================================
# 4. PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* Sidebar */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(150,150,160,0.15);
}


/* Metrics */
[data-testid="stMetric"] {
    border: 1px solid rgba(130,130,150,0.18);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    background: rgba(255,255,255,0.025);
}


/* Buttons */
.stButton > button {
    min-height: 48px;
    border-radius: 12px;
    font-weight: 600;
}


/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 14px;
}


/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
}


/* Main title */
h1 {
    letter-spacing: -0.8px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    return OllamaEmbeddings(
        model="all-minilm:33m",
        base_url="http://127.0.0.1:11434"
    )


embeddings = load_embeddings()


# ============================================================
# 6. LOAD PERSISTENT CHROMA DATABASE
# ============================================================

@st.cache_resource
def load_vectorstore():

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )


vectorstore = load_vectorstore()


# Read the real number of indexed chunks.
chunk_count = vectorstore._collection.count()


# ============================================================
# 7. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)


# ============================================================
# 8. LOAD LOCAL LLM
# ============================================================

@st.cache_resource
def load_llm():

    return OllamaLLM(
        model="llama3.2:3b",
        base_url="http://127.0.0.1:11434",
        temperature=0
    )


llm = load_llm()


# ============================================================
# 9. GROUNDED RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an AI course material assistant.

Answer the user's question using ONLY the retrieved course material.

Follow these rules:

1. Use only information contained in the provided context.

2. Do not invent facts.

3. Explain the answer clearly and in simple language.

4. Use short paragraphs or bullet points when useful.

5. Do NOT write filenames.

6. Do NOT write page numbers.

7. Do NOT generate citations yourself.

8. The application will display the retrieved sources separately.

If the answer is not available in the context, say exactly:

"The answer is not available in the provided course materials."


Context:

{context}


Question:

{question}


Answer:
"""
)


# ============================================================
# 10. FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_docs(docs):
    """
    Clean PDF text before sending it to the LLM.
    """

    chunks = []

    for doc in docs:

        # Remove unusual PDF spaces and line breaks.
        clean_text = " ".join(
            doc.page_content.split()
        )

        chunks.append(clean_text)

    return "\n\n".join(chunks)


# ============================================================
# 11. GET RELIABLE SOURCES FROM METADATA
# ============================================================

def get_retrieved_sources(docs):
    """
    Source names and page numbers come directly
    from Chroma metadata, not from the LLM.
    """

    sources = []

    for doc in docs:

        # Real PDF filename.
        filename = Path(
            doc.metadata["source"]
        ).name

        # LangChain stores page numbers starting from 0.
        page = (
            doc.metadata.get("page", 0)
            + 1
        )

        citation = (
            f"{filename} — Page {page}"
        )

        # Avoid duplicate citations.
        if citation not in sources:
            sources.append(citation)

    return sources


# ============================================================
# 12. INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# 13. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 AI Course Intelligence")

    st.caption(
        "Developed by **Ruturaj Mokashi**"
    )

    st.divider()


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.subheader("About")

    st.write(
        "A local Retrieval-Augmented Generation assistant "
        "that searches multiple AI course documents before "
        "generating an answer."
    )


    st.divider()


    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    st.subheader("Knowledge Base")

    st.write(
        f"📚 **{DOCUMENT_COUNT} PDF documents**"
    )

    st.write(
        f"📄 **{PAGE_COUNT:,} pages**"
    )

    st.write(
        f"🧩 **{chunk_count:,} searchable chunks**"
    )

    st.write(
        "🔎 **Top 4 semantic retrieval**"
    )


    st.divider()


    # --------------------------------------------------------
    # AI STACK
    # --------------------------------------------------------

    st.subheader("AI Stack")


    st.caption("Language Model")

    st.code(
        "Llama 3.2:3b"
    )


    st.caption("Embedding Model")

    st.code(
        "all-minilm:33m"
    )


    st.caption("Vector Database")

    st.code(
        "Chroma"
    )


    st.caption("Framework")

    st.code(
        "LangChain + Streamlit"
    )


    st.caption("Processing")

    st.code(
        "100% Local"
    )


    st.divider()


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 14. MAIN HEADER
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

Explore **Large Language Models, Prompt Engineering, Hugging Face,
AI Agents and Agentic Design Patterns** across multiple course documents.

Ask questions and receive answers grounded in retrieved course material.
"""
    )

    st.caption(
        "RAG  •  Multi-Document Search  •  LangChain  •  "
        "Ollama  •  Chroma  •  Local AI"
    )


# ============================================================
# 15. PROJECT METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        label="Documents",
        value=DOCUMENT_COUNT
    )


with metric2:

    st.metric(
        label="Pages",
        value=f"{PAGE_COUNT:,}"
    )


with metric3:

    st.metric(
        label="Knowledge Chunks",
        value=f"{chunk_count:,}"
    )


with metric4:

    st.metric(
        label="Retrieved Context",
        value="Top 4"
    )


# ============================================================
# 16. TOPIC EXPLORER — REAL TABS
# ============================================================

st.subheader(
    "💡 Explore the Knowledge Base"
)

st.caption(
    "Switch between topics and ask a suggested question."
)


topic_tabs = st.tabs(
    [
        "🧠 Large Language Models",
        "✍️ Prompt Engineering",
        "🤖 AI Agents",
        "🔗 Hugging Face",
        "⚙️ Agentic Design",
        "📚 LLM Applications"
    ]
)


selected_question = None


# ============================================================
# TAB 1 — LARGE LANGUAGE MODELS
# ============================================================

with topic_tabs[0]:

    st.markdown(
        """
### 🧠 Large Language Models

Explore how Large Language Models understand text,
generate responses and power modern AI applications.
"""
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "What is a Large Language Model?",
            key="llm_question_1",
            use_container_width=True
        ):

            selected_question = (
                "What is a Large Language Model "
                "and how does it work?"
            )


    with col2:

        if st.button(
            "How do LLMs generate text?",
            key="llm_question_2",
            use_container_width=True
        ):

            selected_question = (
                "How do Large Language Models generate text?"
            )


# ============================================================
# TAB 2 — PROMPT ENGINEERING
# ============================================================

with topic_tabs[1]:

    st.markdown(
        """
### ✍️ Prompt Engineering

Learn how instructions and prompts influence AI model
responses and how prompts can be improved.
"""
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "What is Prompt Engineering?",
            key="prompt_question_1",
            use_container_width=True
        ):

            selected_question = (
                "What is prompt engineering "
                "and why is it important?"
            )


    with col2:

        if st.button(
            "What makes a good prompt?",
            key="prompt_question_2",
            use_container_width=True
        ):

            selected_question = (
                "What makes an effective prompt?"
            )


# ============================================================
# TAB 3 — AI AGENTS
# ============================================================

with topic_tabs[2]:

    st.markdown(
        """
### 🤖 AI Agents

Explore AI systems that can reason, use tools,
make decisions and perform multi-step tasks.
"""
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "What are AI Agents?",
            key="agent_question_1",
            use_container_width=True
        ):

            selected_question = (
                "What are AI agents and "
                "how do agentic systems work?"
            )


    with col2:

        if st.button(
            "What can AI Agents do?",
            key="agent_question_2",
            use_container_width=True
        ):

            selected_question = (
                "What tasks can AI agents perform?"
            )


# ============================================================
# TAB 4 — HUGGING FACE
# ============================================================

with topic_tabs[3]:

    st.markdown(
        """
### 🔗 Hugging Face

Explore pretrained models, Transformers, pipelines
and the Hugging Face ecosystem.
"""
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "What is Hugging Face?",
            key="hf_question_1",
            use_container_width=True
        ):

            selected_question = (
                "What is Hugging Face "
                "and how is it used with LLMs?"
            )


    with col2:

        if st.button(
            "What are Hugging Face pipelines?",
            key="hf_question_2",
            use_container_width=True
        ):

            selected_question = (
                "What are Hugging Face pipelines "
                "and how are they used?"
            )


# ============================================================
# TAB 5 — AGENTIC DESIGN PATTERNS
# ============================================================

with topic_tabs[4]:

    st.markdown(
        """
### ⚙️ Agentic Design Patterns

Explore common patterns for building intelligent
AI-agent workflows and autonomous systems.
"""
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "Main Agentic Design Patterns",
            key="design_question_1",
            use_container_width=True
        ):

            selected_question = (
                "What are the main agentic design patterns?"
            )


    with col2:

        if st.button(
            "How are Agentic Patterns used?",
            key="design_question_2",
            use_container_width=True
        ):

            selected_question = (
                "How are agentic design patterns "
                "used in AI applications?"
            )


# ============================================================
# TAB 6 — LLM APPLICATIONS
# ============================================================

with topic_tabs[5]:

    st.markdown(
        """
### 📚 LLM Applications

Explore practical business and technical uses
of Large Language Models.
"""
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "Common LLM Applications",
            key="application_question_1",
            use_container_width=True
        ):

            selected_question = (
                "What are common real-world applications "
                "of Large Language Models?"
            )


    with col2:

        if st.button(
            "How are LLMs used in business?",
            key="application_question_2",
            use_container_width=True
        ):

            selected_question = (
                "How are Large Language Models "
                "used in businesses?"
            )


# ============================================================
# 17. CHAT AREA
# ============================================================

st.divider()


st.subheader(
    "💬 Ask the Course Assistant"
)


st.caption(
    "Answers are generated only from the indexed course documents."
)


# ============================================================
# 18. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 19. USER INPUT
# ============================================================

typed_question = st.chat_input(
    "Ask anything about the AI and LLM course materials..."
)


# A manually typed question has priority.
# Otherwise use the question selected from a topic tab.
question = (
    typed_question
    if typed_question
    else selected_question
)


# ============================================================
# 20. PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # DISPLAY USER QUESTION
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # Save user question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the knowledge base..."
        ):

            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            retrieved_docs = (
                retriever.invoke(
                    question
                )
            )


            # ------------------------------------------------
            # PREPARE CONTEXT
            # ------------------------------------------------

            context = format_docs(
                retrieved_docs
            )


            # ------------------------------------------------
            # BUILD PROMPT
            # ------------------------------------------------

            final_prompt = (
                prompt.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )
            )


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            answer = llm.invoke(
                final_prompt.to_string()
            )


            # ------------------------------------------------
            # GET REAL SOURCES
            # ------------------------------------------------

            retrieved_sources = (
                get_retrieved_sources(
                    retrieved_docs
                )
            )


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # DISPLAY SOURCES
        # ----------------------------------------------------

        st.markdown(
            "#### 📚 Retrieved Sources"
        )


        for source in retrieved_sources:

            st.markdown(
                f"- `{source}`"
            )


        # ----------------------------------------------------
        # DISPLAY RETRIEVED CONTEXT
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Retrieved Context"
        ):

            for index, doc in enumerate(
                retrieved_docs,
                start=1
            ):

                # Real source filename.
                filename = Path(
                    doc.metadata["source"]
                ).name


                # Human-readable page number.
                page = (
                    doc.metadata.get(
                        "page",
                        0
                    )
                    + 1
                )


                # Clean the PDF text.
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


    # ========================================================
    # 21. SAVE ASSISTANT ANSWER
    # ========================================================

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
            "content": saved_answer
        }
    )


# ============================================================
# 22. FOOTER
# ============================================================

st.divider()


st.caption(
    "AI Course Intelligence • Multi-Document RAG System"
)


st.caption(
    "Developed by Ruturaj Mokashi"
)


st.caption(
    "LangChain • Ollama • Chroma • Streamlit • Local AI"
)