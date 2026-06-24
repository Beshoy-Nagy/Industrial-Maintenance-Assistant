from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

vectorstore = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

query = input("Ask a question: ")

docs = vectorstore.similarity_search(
    query,
    k=3
)

print("\nRESULTS:\n")

for i, doc in enumerate(docs, start=1):

    print(f"\n--- Result {i} ---\n")

    print(doc.page_content[:1000])