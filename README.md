# 🤖 Agentic AI RAG Assistant

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that allows users to ask questions about an **Agentic AI eBook**.

The application uses **Streamlit** for the user interface, **LangChain** for document processing and retrieval, **Hugging Face embeddings** for semantic search, **InMemoryVectorStore** for vector storage, **Groq** for LLM inference, and **LangGraph** to build the RAG workflow.

---

## 🚀 Features

* 📚 Ask questions about an Agentic AI eBook
* 🔎 Semantic document retrieval using embeddings
* 🧩 Automatic PDF document chunking
* 🧠 Retrieval-Augmented Generation (RAG)
* ⚡ Fast LLM inference using Groq
* 🤖 `openai/gpt-oss-20b` model
* 🔗 LangGraph-based RAG workflow
* 💬 Chat history using Streamlit session state
* 📊 Displays retrieved chunks and relevance scores
* 📝 Shows the raw retrieval distance for each chunk
* 💾 Caches the vector store and LLM for better performance
* 🛡️ Prevents the LLM from generating information outside the provided document

---

## 🏗️ Architecture

The application follows this workflow:

```text
                 ┌───────────────────┐
                 │   Agentic AI PDF  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    PDF Loader     │
                 │   PyPDFLoader     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   Text Splitter   │
                 │ Chunk: 1000       │
                 │ Overlap: 200      │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Hugging Face      │
                 │ Embeddings        │
                 │ MiniLM-L6-v2      │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ InMemoryVectorStore│
                 └─────────┬─────────┘
                           │
                    User Question
                           │
                           ▼
                 ┌───────────────────┐
                 │    Retrieval      │
                 │     Top 8 Chunks  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  Context Creation │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    Groq LLM       │
                 │ gpt-oss-20b       │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │      Answer       │
                 └───────────────────┘
```

---

## 🧠 RAG Workflow

The RAG pipeline is implemented using **LangGraph**.

### 1. Retrieve Node

The user's question is converted into an embedding and compared against the document embeddings.

The application retrieves the **8 most relevant chunks**:

```python
results = vector_store.similarity_search_with_score(
    query=state.question,
    k=8
)
```

---

### 2. Create Context Node

The retrieved chunks are combined into a single context:

```python
state.context = "\n\n".join(
    doc.page_content
    for doc in state.documents
)
```

This context is then provided to the LLM.

---

### 3. Generate Answer Node

The Groq LLM receives:

* Retrieved context
* User question
* Instructions to avoid hallucination

The model is instructed to answer only using information available in the eBook.

If sufficient information is not available, it responds:

```text
I cannot answer based on the provided document.
```

---

## 🛠️ Tech Stack

| Technology                     | Purpose                         |
| ------------------------------ | ------------------------------- |
| Python                         | Programming language            |
| Streamlit                      | Web application UI              |
| LangChain                      | RAG and document processing     |
| LangGraph                      | RAG workflow orchestration      |
| PyPDFLoader                    | PDF document loading            |
| RecursiveCharacterTextSplitter | Text chunking                   |
| Hugging Face                   | Text embeddings                 |
| all-MiniLM-L6-v2               | Embedding model                 |
| InMemoryVectorStore            | Vector storage                  |
| Groq                           | LLM inference                   |
| GPT-OSS-20B                    | Language model                  |
| Pydantic                       | State validation                |
| python-dotenv                  | Environment variable management |

---

## 📁 Project Structure

Recommended project structure:

```text
agentic-ai-rag-assistant/
│
├── app.py
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/agentic-ai-rag-assistant.git
```

Move into the project directory:

```bash
cd agentic-ai-rag-assistant
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

The application loads the API key using:

```python
from dotenv import load_dotenv

load_dotenv()
```

### Getting a Groq API Key

Create an API key from the Groq developer platform and add it to your `.env` file.

**Important:** Never upload your `.env` file or expose your API key on GitHub.

---

## 📦 Requirements

Your `requirements.txt` can contain:

```text
streamlit
python-dotenv
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
langgraph
pydantic
pypdf
sentence-transformers
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 💬 Example Questions

After starting the application, you can ask questions such as:

```text
What is Agentic AI?
```

```text
What are AI agents?
```

```text
What is the difference between an AI agent and a chatbot?
```

```text
What are the components of an AI agent?
```

```text
How do AI agents use tools?
```

```text
What is the role of memory in AI agents?
```

The assistant will retrieve relevant sections from the eBook and generate an answer based on those sections.

---

## 🔍 Retrieval System

The application uses:

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The eBook is divided into chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

### Chunk Configuration

| Parameter        |            Value |
| ---------------- | ---------------: |
| Chunk Size       |  1000 characters |
| Chunk Overlap    |   200 characters |
| Retrieved Chunks |                8 |
| Embedding Model  | all-MiniLM-L6-v2 |

The overlap helps preserve contextual information between neighboring chunks.

---

## 📊 Retrieved Context

The application provides a sidebar showing the retrieved chunks.

For each chunk, it displays:

* Chunk number
* Calculated relevance percentage
* Relevance progress bar
* Retrieved text
* Raw retrieval distance

Example:

```text
📚 Retrieved Context

8 chunks retrieved

Chunk 1 | 82.4% relevance
████████████████████

Retrieved document content...

Raw retrieval distance: 0.176
```

This makes the retrieval process more transparent and helps users understand which parts of the document were used to generate the answer.

---

## 🔗 LangGraph Workflow

The RAG pipeline is represented as a graph:

```text
START
  │
  ▼
Retrieve Node
  │
  ▼
Create Context Node
  │
  ▼
Generate Node
  │
  ▼
 END
```

The graph is created using:

```python
graph = StateGraph(RagState)
```

The nodes are:

```text
retrieve_node
create_context_node
generate_node
```

---

## 🧠 RAG State

The application maintains the following state:

```python
class RagState(BaseModel):

    question: str

    documents: list

    scores: list

    context: str

    answer: str
```

### State Fields

| Field       | Description                |
| ----------- | -------------------------- |
| `question`  | User's question            |
| `documents` | Retrieved document chunks  |
| `scores`    | Retrieval scores           |
| `context`   | Combined retrieved context |
| `answer`    | Generated response         |

---

## ⚡ Performance Optimization

The application uses Streamlit's resource caching:

```python
@st.cache_resource
```

It is applied to:

```python
load_vector_store()
```

and:

```python
load_llm()
```

This prevents the vector store and LLM from being unnecessarily recreated on every Streamlit interaction.

---

## 🛡️ Hallucination Control

The application uses a strict prompt:

```text
Answer the question ONLY using the context provided below.

If the context does not contain enough information, say:

"I cannot answer based on the provided document."

Do not invent information.
```

This helps constrain the model to the retrieved eBook content rather than relying on unrelated knowledge.

---

## 🔐 Security

Do not commit your API key to GitHub.

Add the following to `.gitignore`:

```text
.env
venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
```

Never write:

```text
GROQ_API_KEY=your_actual_key
```

directly inside your Python source code.

---

## 🚀 Future Improvements

Possible improvements for future versions:

* [ ] Add document upload functionality
* [ ] Support multiple PDFs
* [ ] Add source citations to answers
* [ ] Add page numbers for retrieved chunks
* [ ] Add streaming responses
* [ ] Add conversation-aware retrieval
* [ ] Add query rewriting
* [ ] Add retrieval quality evaluation
* [ ] Add reranking of retrieved chunks
* [ ] Add persistent vector database
* [ ] Add authentication
* [ ] Deploy the application online
* [ ] Add configurable retrieval `k`
* [ ] Add document metadata filtering

---

## 🌐 Deployment

The application can be deployed using platforms that support Streamlit applications.

Before deployment, make sure:

1. `requirements.txt` is present.
2. The PDF is included in the repository or provided through another storage mechanism.
3. The Groq API key is configured securely through the deployment platform's secrets/environment-variable system.
4. `.env` is not committed to GitHub.

---

## 📌 Important Notes

### InMemoryVectorStore

This project currently uses:

```python
InMemoryVectorStore
```

Therefore, the vector database exists only while the application is running.

When the application restarts, the PDF needs to be processed and embedded again.

For larger applications, a persistent vector database such as Chroma, FAISS, Qdrant, or another vector database can be considered.

### Local Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

for embeddings.

This means the embedding model is used locally rather than requiring an embedding API.

### LLM

The application uses Groq for LLM inference:

```python
ChatGroq(
    model="openai/gpt-oss-20b"
)
```

The API key is therefore required for generating answers.

---

## 🎯 Project Objective

The main objective of this project is to demonstrate how **Retrieval-Augmented Generation (RAG)** can be combined with **LangGraph** to build a document-based AI assistant.

The system demonstrates the complete pipeline:

```text
PDF
 ↓
Document Loading
 ↓
Text Splitting
 ↓
Embeddings
 ↓
Vector Store
 ↓
Semantic Retrieval
 ↓
Context Construction
 ↓
LLM
 ↓
Grounded Answer
```

---

## 👨‍💻 Author

**Shashidhar Kokku**

B.Tech – Computer Science and Data Science

Sri Indu College of Engineering and Technology

---

## ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and demonstration purposes.
