# 🏦 AI Banking Knowledge Assistant — RAG + LLM

A Retrieval-Augmented Generation (RAG) application that answers banking-related questions using information retrieved from a curated banking knowledge base.

The system combines semantic search, OpenAI embeddings, ChromaDB and a Large Language Model (LLM) to generate answers grounded in retrieved source material rather than relying only on the model's general knowledge.

🔗 **Live Demo:** https://vivek-rag-ai-assistant-project.streamlit.app/

---

## 🚀 Project Overview

Large Language Models can generate convincing answers even when the required information is not available to them.

This project explores how Retrieval-Augmented Generation can improve grounding by retrieving relevant information from a controlled knowledge base before generating an answer.

The application can:

- Load information from PDF and web sources
- Clean and preprocess source text
- Split documents into overlapping chunks
- Generate vector embeddings
- Store embeddings in ChromaDB
- Perform semantic similarity search
- Filter weak retrievals using a relevance threshold
- Generate answers using retrieved context
- Display source names and PDF page numbers
- Show retrieval scores and retrieved text
- Refuse unsupported/out-of-domain questions

---

## 🧠 How It Works

```text
User Question
      ↓
OpenAI Embedding
      ↓
ChromaDB Vector Search
      ↓
Top-K Similarity Retrieval
      ↓
Relevance Threshold
      ↓
Relevant Banking Context
      ↓
LLM Generation
      ↓
Grounded Answer + Sources
```

Instead of sending only the user's question to the LLM, the application first searches the vector database for relevant banking information.

Only retrieved documents that pass the relevance threshold are supplied as context to the language model.

If the system cannot find sufficiently relevant information, it returns a fallback response instead of answering from outside the knowledge base.

---

## 📚 Knowledge Base

The current knowledge base contains information from Financial Conduct Authority (FCA) material.

### FCA TR17/1 — Customer Understanding in Retail Banking

A Financial Conduct Authority thematic review covering customer understanding within retail banks and building societies.

The PDF is processed page-by-page so page metadata can be retained and displayed with generated answers.

### FCA — Fraudulent Payments

FCA consumer information covering fraudulent payments and what consumers can do when they believe they have been affected by payment fraud or scams.

The webpage is retrieved and cleaned before being added to the vector database.

---

## 🔍 Document Ingestion

The ingestion pipeline supports both PDF and web sources.

### PDF Processing

PDF pages are loaded using `PyPDFLoader`.

Metadata is attached to each page, including:

- Source type
- Source name
- Original filename
- Page number
- Chunk number

### Web Processing

Web content is retrieved using `requests` and parsed with `BeautifulSoup`.

Scripts, navigation, headers, footers and styles are removed before extracting the main content. Whitespace is then normalised before chunking.

---

## ✂️ Chunking Strategy

Documents are split using LangChain's `RecursiveCharacterTextSplitter`.

```text
Chunk size: 700 characters
Chunk overlap: 120 characters
```

Overlap helps preserve context when information spans chunk boundaries.

The current knowledge base produces approximately **61 chunks**.

---

## 🔢 Embeddings

Embeddings are generated using:

```text
text-embedding-3-small
```

Each text chunk is transformed into a numerical vector representing its semantic meaning. This allows retrieval based on semantic similarity rather than only exact keyword matching.

---

## 🗄️ Vector Database

The application uses **ChromaDB** as its vector store.

Chroma stores document text, vector embeddings and associated source/page metadata.

The persisted vector database is included with the deployed application so the knowledge base does not need to be re-embedded whenever the Streamlit application starts.

---

## 🔎 Retrieval

For each user question:

1. The question is converted into an embedding.
2. ChromaDB performs semantic similarity search.
3. The top matching chunks are returned.
4. A relevance threshold filters weak matches.
5. Accepted chunks are passed to the LLM.

Current configuration:

```text
Top-K: 4
Relevance threshold: 0.30
```

The threshold is a project-specific value selected after inspecting retrieval behaviour and is not assumed to be universally optimal.

---

## 🛡️ Out-of-Domain Handling

Vector databases can return the nearest available vectors even when a question is unrelated to the knowledge base.

Therefore:

> **Nearest result does not necessarily mean relevant result.**

The application applies a relevance threshold before generation.

For example, an unrelated question such as:

```text
What is the price of Bitcoin today?
```

is rejected when the retrieved banking information does not meet the relevance requirement.

The application then returns:

```text
I don't have enough information in the provided banking documents.
```

This helps reduce unsupported answers.

---

## 🤖 Answer Generation

Relevant retrieved chunks are combined into structured context and supplied to the LLM.

The prompt instructs the model to:

- Use only retrieved banking context
- Avoid outside knowledge
- Avoid inventing facts
- Combine useful information from multiple passages
- Provide concise answers
- Return the fallback response when the context cannot support an answer

Current generation model:

```text
gpt-5.6-luna
```

---

## 📑 Source Attribution

Generated answers display the sources associated with accepted retrieved documents.

PDF results can include the original page number, for example:

```text
FCA TR17/1 – Customer Understanding in Retail Banking — Page 8
```

The application also includes a **View retrieval details** section showing:

- Relevance score
- Source
- Page number where available
- URL where available
- Retrieved text

This makes the retrieval process inspectable and easier to debug.

---

## 🧪 Retrieval Evaluation

A small retrieval evaluation set was created to test whether the system retrieves relevant sources and rejects unsupported questions.

The initial evaluation contained **8 questions** covering:

- Fraudulent payments
- Unauthorised payments
- APP fraud/reimbursement
- Customer understanding in retail banking
- Out-of-domain questions

Result:

```text
Passed: 8/8
Retrieval/refusal decision accuracy: 100%
```

This result applies only to the current **8-question evaluation set** and does not mean that the overall RAG system is 100% accurate.

---

## 🧩 Challenges & Solutions

### PDF Extraction Failure

One attempted FCA PDF did not extract into usable text. Instead of applying unnecessary OCR, the project switched to direct ingestion from the corresponding FCA webpage.

### Web Text Formatting

Initial webpage extraction produced joined words and noisy formatting. The ingestion pipeline was changed to use `requests` and `BeautifulSoup` with space-separated extraction and whitespace normalisation.

### Relevance Threshold Tuning

An earlier threshold rejected legitimate FCA PDF chunks. Retrieval scores were inspected and the threshold was adjusted before running the evaluation set.

### Cloud Deployment

Deployment required resolving issues involving:

- Project path handling
- Dependency installation
- Streamlit dependency-file precedence
- Environment secrets
- Persisted ChromaDB paths
- Streamlit session state during redeployment

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| LangChain | RAG pipeline components |
| OpenAI API | Embeddings and LLM generation |
| ChromaDB | Vector database |
| BeautifulSoup | Web content parsing |
| Requests | Web retrieval |
| PyPDF | PDF processing |
| Streamlit | User interface and deployment |
| Git / GitHub | Version control |

---

## 📁 Project Structure

```text
banking-rag-assistant/
│
├── data/
│   └── pdfs/
│       └── tr17-1.pdf
│
├── chroma_db/
│
├── app.py
├── ingest.py
├── rag.py
├── rag_engine.py
├── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Main Files

**`ingest.py`**  
Loads source documents, cleans text, creates chunks, generates embeddings and builds the Chroma vector database.

**`rag_engine.py`**  
Handles retrieval, relevance filtering, context construction, LLM generation and source extraction.

**`app.py`**  
Provides the Streamlit chat interface, conversation state, source display and retrieval inspection.

**`evaluate.py`**  
Runs the retrieval evaluation set without invoking the generation model unnecessarily.

---

## 💻 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/vivekavardhan484/banking-ai-portfolio-projects.git
```

### 2. Enter the project

```bash
cd banking-ai-portfolio-projects/banking-rag-assistant
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate it on Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the OpenAI API key

Create a `.env` file inside the project directory:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file should never be committed to Git.

### 7. Build the vector database

```bash
python ingest.py
```

### 8. Run the application

```bash
python -m streamlit run app.py
```

---

## 🔐 Security

API credentials are not stored in the source code or committed to GitHub.

Local development uses environment variables loaded from `.env`, while the deployed Streamlit application uses environment secrets.

---

## ⚠️ Limitations

This is an educational portfolio project and not a production banking system.

Current limitations include:

- Small curated knowledge base
- Limited retrieval evaluation dataset
- Retrieval threshold tuned on a small number of examples
- Answers depend on source-document quality and coverage
- No automated continuous knowledge-base updates
- No authentication or user-specific banking information
- No production-scale observability or evaluation framework

**This application does not provide financial advice.**

---

## 🔮 Future Improvements

- Expand the banking knowledge base
- Add automated RAG evaluation
- Compare different chunking and retrieval strategies
- Add reranking
- Introduce hybrid keyword + vector retrieval
- Add FastAPI endpoints
- Containerise with Docker
- Add automated tests
- Add logging and monitoring

---

## 🎯 What I Learned

This project provided practical experience with:

- Retrieval-Augmented Generation (RAG)
- LLM application development
- Semantic search
- Vector embeddings
- Vector databases
- Document ingestion and text cleaning
- Chunking and metadata management
- Prompt design
- Retrieval evaluation
- Hallucination reduction strategies
- Streamlit deployment
- Cloud debugging
- API secret management

> **Retrieval quality is as important as generation quality.**

---

## 👨‍💻 Author

**Vivek Kothapalli**

MSc Artificial Intelligence with Business Strategy  
Aston University, Birmingham

GitHub: https://github.com/vivekavardhan484

LinkedIn: https://www.linkedin.com/in/vivek-kothapalli