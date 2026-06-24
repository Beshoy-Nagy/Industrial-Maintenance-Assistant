\# ⚙️ Industrial Maintenance Assistant



An AI-powered Industrial Maintenance Assistant built with \*\*RAG (Retrieval-Augmented Generation)\*\*, \*\*LangChain\*\*, \*\*FAISS\*\*, \*\*Ollama\*\*, and \*\*Streamlit\*\*.



The system allows engineers and technicians to upload industrial manuals, maintenance guides, troubleshooting documents, and technical catalogs, then ask natural language questions and receive accurate answers grounded in the uploaded documentation.



\---



\## 🚀 Features



\### 📚 Knowledge Base Management



\* Upload PDF manuals directly from the UI

\* Automatic PDF parsing and chunking

\* Incremental indexing for newly uploaded documents

\* Full database rebuild option

\* Metadata tracking for document names and page numbers



\### 🔍 Intelligent Retrieval



\* Semantic search using vector embeddings

\* FAISS vector database for fast retrieval

\* Top-K document retrieval

\* Relevance filtering using similarity thresholds



\### 🤖 AI-Powered Question Answering



\* Local LLM inference using Ollama-Qwen3:4B

\* Context-aware responses

\* Answers generated only from retrieved documentation

\* Reduced hallucinations through RAG architecture



\### 📄 Source Attribution



\* Displays source documents used

\* Shows exact page references

\* Improves traceability and reliability



\### 🔒 Fully Local Deployment



\* No cloud dependency

\* Documents remain on-premise

\* Suitable for industrial environments with privacy requirements



\---



\## 🏗️ System Architecture



User Question

↓

FAISS Similarity Search

↓

Relevant Document Chunks

↓

Context Construction

↓

Qwen3 LLM (Ollama)

↓

Grounded Response + Sources



\---



\## 🛠️ Technology Stack



\### Frontend



\* Streamlit



\### LLM



\* Qwen 3 (Ollama)



\### Embeddings



\* Nomic Embed Text



\### RAG Components



\* LangChain

\* FAISS



\### Document Processing



\* PyPDF

\* Recursive Character Text Splitter



\### Backend



\* Python



\---



\## 📂 Project Structure



```text

Industrial-Maintenance-Assistant/

│

├── app.py

├── ingest.py

├── config.py

├── prompts.py

├── requirements.txt

├── README.md

│

├── data/

│   └── pdfs/

│

├── vector\_db/

│

└── assets/

```



\## ⚡ Installation



\### Clone Repository



```bash

git clone https://github.com/yourusername/Industrial-Maintenance-Assistant.git



cd Industrial-Maintenance-Assistant

```



\### Create Virtual Environment



```bash

python -m venv .venv

```



\### Activate Environment



Windows:



```bash

.venv\\Scripts\\activate

```



Linux/Mac:



```bash

source .venv/bin/activate

```



\### Install Dependencies



```bash

pip install -r requirements.txt

```



\---



\## 🦙 Install Ollama Models



Pull the embedding model:



```bash

ollama pull nomic-embed-text

```



Pull the language model:



```bash

ollama pull qwen3:4b

```



Verify installation:



```bash

ollama list

```



\---



\## ▶️ Run Application



```bash

streamlit run app.py

```



\---



\## 📖 Usage



\### Upload Documents



1\. Open the application.

2\. Upload industrial manuals or maintenance guides.

3\. Documents are automatically indexed.



\### Ask Questions



Example:



```text

What are the causes of excessive bearing vibration?

```



```text

How can motor overheating be diagnosed?

```



```text

What is the service factor of catalog number 00118ST3QIE143T-W22?

```



```text

What are the recommended troubleshooting steps for a W22 motor?

```



\---



\## 📊 Example Capabilities



The assistant can answer questions related to:



\* Electric Motors

\* Bearings

\* Vibration Analysis

\* Predictive Maintenance

\* Industrial Troubleshooting

\* Reliability Engineering

\* Rotating Equipment

\* Mechanical Systems

\* Maintenance Manuals

\* Technical Catalogs



\---



\## 🎯 Future Improvements



\* Hybrid Search (BM25 + Vector Search)

\* Metadata Filtering

\* OCR Support for Scanned PDFs

\* Multi-document Collections

\* Conversation Memory

\* Citation Highlighting

\* PDF Preview Integration

\* Docker Deployment



\---



\## 👨‍💻 Author



\*\*Beshoy Nagy\*\*



AI Engineer | Machine Learning Engineer | Computer Vision Enthusiast



LinkedIn:

( https://www.linkedin.com/in/beshoy-nagy-4b33a6212/ )



GitHub:

( https://github.com/Beshoy-Nagy )



