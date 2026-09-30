import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# Page Header
# --------------------------------------------------

st.title("🤖 Agentic AI RAG Assistant")
st.caption("Ask questions about the Agentic AI eBook")


# --------------------------------------------------
# Load PDF + Create Vector Store
# --------------------------------------------------

@st.cache_resource
def load_vector_store():

    loader = PyPDFLoader(
        "data/Ebook-Agentic-AI.pdf"
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embedding
    )

    return vector_store


# --------------------------------------------------
# Load Groq LLM
# --------------------------------------------------

@st.cache_resource
def load_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-20b"
    )

    return llm


vector_store = load_vector_store()
llm = load_llm()


# --------------------------------------------------
# RAG State
# --------------------------------------------------

class RagState(BaseModel):

    question: str = Field(
        description="User question"
    )

    documents: list = Field(
        default_factory=list
    )

    scores: list = Field(
        default_factory=list
    )

    context: str = Field(
        default=""
    )

    answer: str = Field(
        default=""
    )


# --------------------------------------------------
# Retrieve Node
# --------------------------------------------------

def retrieve_node(state: RagState) -> RagState:

    results = vector_store.similarity_search_with_score(
        query=state.question,
        k=8
    )

    documents = []
    scores = []

    for doc, score in results:

        documents.append(doc)
        scores.append(score)

    state.documents = documents
    state.scores = scores

    return state


# --------------------------------------------------
# Create Context Node
# --------------------------------------------------

def create_context_node(state: RagState) -> RagState:

    state.context = "\n\n".join(
        doc.page_content
        for doc in state.documents
    )

    return state


# --------------------------------------------------
# Generate Answer Node
# --------------------------------------------------

def generate_node(state: RagState) -> RagState:

    prompt = f"""
You are a strict assistant.

Answer the question ONLY using the context provided below.

If the context does not contain enough information, say:

"I cannot answer based on the provided document."

Do not invent information.

Context:
{state.context}

Question:
{state.question}

Answer:
"""

    response = llm.invoke(prompt)

    state.answer = response.content

    return state


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

graph = StateGraph(RagState)


graph.add_node(
    "retrieve_node",
    retrieve_node
)

graph.add_node(
    "create_context_node",
    create_context_node
)

graph.add_node(
    "generate_node",
    generate_node
)


graph.add_edge(
    START,
    "retrieve_node"
)

graph.add_edge(
    "retrieve_node",
    "create_context_node"
)

graph.add_edge(
    "create_context_node",
    "generate_node"
)

graph.add_edge(
    "generate_node",
    END
)


app = graph.compile()


# --------------------------------------------------
# Chat History
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# Display Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

question = st.chat_input(
    "Ask something about the Agentic AI eBook..."
)


if question:

    # ----------------------------------------------
    # Display User Question
    # ----------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # ----------------------------------------------
    # Run RAG
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching the eBook..."):

            result = app.invoke({
                "question": question
            })

            answer = result["answer"]

            retrieved_docs = result["documents"]

            scores = result["scores"]

        st.markdown(answer)


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


    # ----------------------------------------------
    # Retrieved Context Sidebar
    # ----------------------------------------------

    with st.sidebar:

        st.subheader("📚 Retrieved Context")

        st.caption(
            f"{len(retrieved_docs)} chunks retrieved"
        )

        for i, doc in enumerate(retrieved_docs):

            score = scores[i]

            # Convert distance to UI relevance
            relevance = max(
                0,
                min(
                    100,
                    (1 - score) * 100
                )
            )

            with st.expander(
                f"Chunk {i + 1} | {relevance:.1f}% relevance"
            ):

                st.progress(
                    int(relevance)
                )

                st.write(
                    doc.page_content
                )

                st.caption(
                    f"Raw retrieval distance: {score:.4f}"
                )