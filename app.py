import os
from collections import defaultdict

import streamlit as st
import ollama

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from prompts import RAG_PROMPT
from config import *
from ingest import build_vector_db
from ingest import add_pdf_to_vector_db


# ==================================
# Page Config
# ==================================

st.set_page_config(
    page_title="Industrial Maintenance Assistant",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Industrial Maintenance Assistant")
st.sidebar.title("Controls")
if "rebuild_message" in st.session_state:

    st.sidebar.success(
        st.session_state["rebuild_message"]
    )

    del st.session_state["rebuild_message"]

# ==================================
# Sidebar
# ==================================

os.makedirs(
    PDF_FOLDER,
    exist_ok=True
)

pdf_count = len([
    f for f in os.listdir(PDF_FOLDER)
    if f.lower().endswith(".pdf")
])

st.sidebar.write(
    f"PDF Files: {pdf_count}"
)

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# ==================================
# Upload PDF
# ==================================

st.sidebar.header("📤 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if uploaded_file.size == 0:

        st.sidebar.error(
            "Uploaded file is empty."
        )

        st.stop()

    save_path = os.path.join(
        PDF_FOLDER,
        uploaded_file.name
    )

    if os.path.exists(save_path):

        st.sidebar.warning(
            f"'{uploaded_file.name}' detected.\n\n"
            " A new document will be indexed automatically. Existing documents will be skipped."
        )

    else:

        with open(save_path, "wb") as f:

            f.write(
                uploaded_file.getbuffer()
            )
        try:
            with st.spinner(
                "Indexing PDF..."
            ):
                if os.path.exists(
                    VECTOR_DB_PATH
                ):

                    stats = add_pdf_to_vector_db(
                        save_path
                    )
                else:

                    stats = build_vector_db()    
        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)
                
            st.cache_resource.clear()
            try:
                st.cache_resource.clear()
            except :
                pass        
            st.sidebar.error(f"Indexing failed: {e}")
            st.stop()    
               
        st.cache_resource.clear()

        st.sidebar.success(
            f"Added Successfully\n\n"
            f"Pages: {stats['pages']}\n"
            f"Chunks: {stats['chunks']}"
        )

        st.rerun()

# ==================================
# Rebuild Vector DB
# ==================================

st.sidebar.divider()

st.sidebar.header(
    "🔄 Vector Database"
)

st.sidebar.info(
    "PDFs are indexed automatically after upload.\n\n"
    "Use Rebuild Vector DB only if:\n\n"
    "1 - Search results seem incorrect\n\n"
    "2 - PDFs were modified manually\n\n"
    "3 - The database becomes corrupted"
)

if st.sidebar.button(
    "🔄 Rebuild Vector DB",
    use_container_width=True
):

    try:

        with st.spinner(
            "Rebuilding Vector Database..."
        ):

            stats = build_vector_db()
            st.cache_resource.clear()
            st.session_state["rebuild_message"] = (
                f"Done! PDFs={stats['pdfs']} | "
                f"Pages={stats['pages']} | "
                f"Chunks={stats['chunks']}"
            )    

        

        st.rerun()

    except Exception as e:

        st.sidebar.error(
            f"Error: {str(e)}"
        )


# ==================================
# System Information
# ==================================

st.sidebar.divider()

st.sidebar.subheader(
    "System Information"
)

st.sidebar.write(
    f"LLM: {LLM_MODEL}"
)

st.sidebar.write(
    f"Embedding: {EMBED_MODEL}"
)

st.sidebar.write(
    f"Top K: {TOP_K}"
)

st.sidebar.write(
    f"Threshold: {SCORE_THRESHOLD}"
)

# ==================================
# Session State
# ==================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==================================
# Load Vector DB
# ==================================

@st.cache_resource
def load_vectorstore():

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


try:

    vectorstore = load_vectorstore()

except Exception:

    st.warning(
        "Vector Database not found.\n\n"
        "Upload PDFs and click Rebuild Vector DB."
    )

    st.stop()


# ==================================
# Render Chat History
# ==================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(
            msg["content"]
        )
        
        if "source_count" in msg:

            st.caption(
                f"Sources Used: {msg['source_count']}"
            )

        if "sources" in msg:

            with st.expander("Sources"):

                pages_by_pdf = defaultdict(set)

                for source, page in msg["sources"]:

                    pdf_name = source
                        

                    pages_by_pdf[pdf_name].add(
                        page 
                    )

                for pdf_name, pages in pages_by_pdf.items():

                    pages_str = ", ".join(
                        map(
                            str,
                            sorted(pages)
                        )
                    )

                    st.markdown(
                        f"📄 **{pdf_name}**\n\n"
                        f"Pages: {pages_str}"
                    )


# ==================================
# User Input
# ==================================

question = st.chat_input(
    "Ask a maintenance question..."
)


# ==================================
# Ask Question
# ==================================

if question and question.strip():
    
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.spinner(
        "Searching knowledge base..."
    ):

        results = (
            vectorstore.similarity_search_with_score(
                question,
                k=TOP_K
            )
        )

        context_parts = []

        sources = set()

        for doc, score in results:

            if score < SCORE_THRESHOLD:

                context_parts.append(
                    doc.page_content
                )

                source = doc.metadata.get(
                    "document_name",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page_number",
                    0
                )

                sources.add(
                    (
                        source,
                        page
                    )
                )

        if len(context_parts) == 0:

            answer = (
                "I could not find sufficient "
                "information in the knowledge base."
            )

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    answer
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": [],
                    "source_count": 0
                }
            )

            st.stop()

        context = "\n\n".join(
            context_parts
        )

        prompt = RAG_PROMPT.format(
            context=context,
            question=question
        )

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response[
            "message"
        ][
            "content"
        ]

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )
        st.caption(
            f"Sources Used: {len(sources)}"
        )
        with st.expander(
            "Sources"
        ):

            pages_by_pdf = defaultdict(set)

            for source, page in sorted(
                sources
            ):

                pdf_name = source

                pages_by_pdf[
                    pdf_name
                ].add(
                    page 
                )

            for pdf_name, pages in pages_by_pdf.items():

                pages_str = ", ".join(
                    map(
                        str,
                        sorted(pages)
                    )
                )

                st.markdown(
                    f"📄 **{pdf_name}**\n\n"
                    f"Pages: {pages_str}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sorted(sources),
            "source_count": len(sources)
        }
    )