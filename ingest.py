import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    PDF_FOLDER,
    VECTOR_DB_PATH,
    EMBED_MODEL
)


# ==================================
# Embeddings
# ==================================

def get_embeddings():

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL
    )

    try:

        embeddings.embed_query(
            "test"
        )

    except Exception as e:

        raise RuntimeError(
            "Embedding model is unavailable. "
            "Make sure Ollama is running and the model is installed."
        ) from e

    return embeddings


# ==================================
# Text Splitter
# ==================================

def get_splitter():

    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


# ==================================
# Metadata Helper
# ==================================

def add_metadata(
    docs,
    pdf_name,
    pdf_path
):

    for doc in docs:

        doc.metadata.update(
            {
                "document_name": pdf_name,
                "file_path": pdf_path,
                "page_number": (
                    doc.metadata.get(
                        "page",
                        0
                    ) + 1
                ),
                "document_type": "pdf"
            }
        )

    return docs


# ==================================
# FULL REBUILD
# ==================================

def build_vector_db():

    if not os.path.exists(
        PDF_FOLDER
    ):

        raise FileNotFoundError(
            f"PDF folder not found: {PDF_FOLDER}"
        )

    pdf_files = [

        f for f in os.listdir(
            PDF_FOLDER
        )

        if f.lower().endswith(
            ".pdf"
        )
    ]

    if not pdf_files:

        raise ValueError(
            "No PDF files found."
        )

    print("=" * 60)
    print("FULL VECTOR DB REBUILD")
    print("=" * 60)

    documents = []

    for file in pdf_files:

        pdf_path = os.path.join(
            PDF_FOLDER,
            file
        )

        print(
            f"Loading: {file}"
        )

        loader = PyPDFLoader(
            pdf_path
        )

        docs = loader.load()

        docs = add_metadata(
            docs,
            file,
            pdf_path
        )

        documents.extend(
            docs
        )

    print(
        f"\nPages Loaded: {len(documents)}"
    )

    splitter = get_splitter()

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Chunks Created: {len(chunks)}"
    )

    embeddings = get_embeddings()

    if os.path.exists(
        VECTOR_DB_PATH
    ):

        print(
            "\nRemoving old Vector DB..."
        )

        shutil.rmtree(
            VECTOR_DB_PATH,
            ignore_errors=True
        )

    print(
        "\nCreating Vector Database..."
    )

    batch_size = 100

    first_batch = min(
        batch_size,
        len(chunks)
    )

    vectorstore = FAISS.from_documents(
        chunks[:first_batch],
        embeddings
    )

    total_batches = (
        len(chunks) +
        batch_size - 1
    ) // batch_size

    current_batch = 1

    for i in range(
        first_batch,
        len(chunks),
        batch_size
    ):

        current_batch += 1

        print(
            f"Batch {current_batch}/{total_batches}"
        )

        batch = chunks[
            i:i + batch_size
        ]

        vectorstore.add_documents(
            batch
        )

    vectorstore.save_local(
        VECTOR_DB_PATH
    )

    print(
        "\nVector DB Created Successfully"
    )

    print("=" * 60)

    return {
        "pdfs": len(pdf_files),
        "pages": len(documents),
        "chunks": len(chunks)
    }


# ==================================
# ADD SINGLE PDF
# ==================================

def add_pdf_to_vector_db(
    pdf_path
):

    if not os.path.exists(
        VECTOR_DB_PATH
    ):

        raise FileNotFoundError(
            "Vector DB does not exist. "
            "Run Full Rebuild first."
        )

    pdf_name = os.path.basename(
        pdf_path
    )

    print(
        f"\nAdding PDF: {pdf_name}"
    )

    loader = PyPDFLoader(
        pdf_path
    )

    documents = loader.load()
    
    total_chars = sum(
        len(doc.page_content)
        for doc in documents
    )

    print(f"Pages: {len(documents)}")
    print(f"Total Characters: {total_chars}")

    documents = add_metadata(
        documents,
        pdf_name,
        pdf_path
    )

    print(f"Pages: {len(documents)}")
    
    splitter = get_splitter()

    chunks = splitter.split_documents(
        documents
    )
    
    print(f"Chunks: {len(chunks)}")
    print("STEP 1")
    embeddings = get_embeddings()   
    print("STEP 2")
    print("STEP 3")
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("STEP 4")
    ###########
    
    batch_size = 50

    for i in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            i:i + batch_size
        ]

        try:
        
            vectorstore.add_documents(
                batch
            )

            print(
                f"Batch {(i // batch_size) + 1}/"
                f"{(len(chunks) + batch_size - 1) // batch_size}"
            )
        except Exception as e:
            print(f"FAILED AT BATCH {(i // batch_size) + 1}")
            print(str(e))
            raise

    ############
    print("STEP 5")
    vectorstore.save_local(
        VECTOR_DB_PATH
    )

    print(
        f"Added {len(chunks)} chunks"
    )

    return {
        "pdf": pdf_name,
        "pages": len(documents),
        "chunks": len(chunks)
    }
 
# ==================================
# MAIN
# ==================================

if __name__ == "__main__":

    stats = build_vector_db()

    print("\nSummary")

    print(
        f"PDFs   : {stats['pdfs']}"
    )

    print(
        f"Pages  : {stats['pages']}"
    )

    print(
        f"Chunks : {stats['chunks']}"
    )
