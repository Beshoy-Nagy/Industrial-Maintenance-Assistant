#This is for testing the LLM query functionality before the main application (GUI),
#It loads the vector database and allows the user to ask questions based on the indexed PDFs.

import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from prompts import RAG_PROMPT
import ollama

# Load Embedding Model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

# Load Vector Database
vectorstore = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("=" * 60)
print("Industrial Maintenance Assistant")
print("Type your question and press Enter.")
print("Type 'bye' to exit.")
print("=" * 60)

while True:

    question = input("\nAsk a question: ").strip()

    if question.lower() == "bye":
        print("\nGoodbye!")
        break

    results = vectorstore.similarity_search_with_score(
        question,
        k=10
    )

    context_parts = []
    sources = set()

    for doc, score in results:

        
        if score < 0.75:

            context_parts.append(
                doc.page_content
            )

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            sources.add(source)

    if len(context_parts) == 0:

        print("\nNo relevant information found in the knowledge base.")
        continue

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a senior industrial maintenance engineer.

Answer ONLY using the provided context.

Rules:
- Use all relevant information.
- If there are multiple causes, list all of them.
- If there are multiple solutions, summarize all of them.
- Be concise and technical.
- Format your answer using Markdown.
- Use headings and bullet points.
- Do not invent information.
- Provide a detailed answer.
- Use all retrieved information.
- Use only information explicitly found in the context.
- Do not infer numerical values.
- Do not generalize beyond the retrieved text.
- If information is uncertain, state that it is uncertain.
- Do not provide a short summary if sufficient information exists.
- If the answer is not found in the context, say:
  "I could not find sufficient information in the knowledge base."

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(
        response["message"]["content"]
    )

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in sorted(sources):
        print("-", source)

    print("\n" + "=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)

    print(f"Chunks Used: {len(context_parts)}")

    for i, (doc, score) in enumerate(results[:3], start=1):

        print(f"\nChunk {i}")
        print(f"Score: {score:.4f}")
        print("-" * 60)

        print(doc.page_content[:400])