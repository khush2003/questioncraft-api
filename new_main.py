import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, TypedDict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel


class ChatOpenRouter(ChatOpenAI):
    """Wrapper for OpenRouter-hosted chat models."""

    def __init__(
        self,
        model_name: str,
        openai_api_key: Optional[str] = None,
        openai_api_base: str = "https://openrouter.ai/api/v1",
        **kwargs,
    ) -> None:
        api_key = openai_api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=api_key,
            model_name=model_name,
            **kwargs,
        )


app = FastAPI(title="QuestionCraft LangGraph QA", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOpenRouter(model_name="meta-llama/llama-3.2-3b-instruct:free", temperature=0)

app.state.vector_store: Optional[FAISS] = None


class UploadResult(BaseModel):
    documents: int
    chunks: int


class QueryPayload(BaseModel):
    query: str
    k: int = 4


class QueryResult(BaseModel):
    answer: str
    sources: list[dict]


class QAState(TypedDict, total=False):
    query: str
    k: int
    documents: List[Document]
    scores: List[float]
    answer: str


def _ensure_vector_store() -> FAISS:
    store = app.state.vector_store
    if store is None:
        raise HTTPException(status_code=400, detail="Upload documents before querying.")
    return store


async def _load_file(upload: UploadFile) -> List[Document]:
    raw_bytes = await upload.read()
    filename = upload.filename or "document"
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(raw_bytes)
            temp_path = Path(tmp.name)
        loader = PyPDFLoader(str(temp_path))
        documents = loader.load()
        temp_path.unlink(missing_ok=True)
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")
        documents = [Document(page_content=text, metadata={"source": filename, "page_number": 1})]

    for doc in documents:
        doc.metadata.setdefault("source", filename)
    return documents


def _format_docs(docs: List[Document]) -> str:
    formatted = []
    for idx, doc in enumerate(docs):
        meta = {
            "source": doc.metadata.get("source"),
            "page_number": doc.metadata.get("page_number"),
            "chunk_index": doc.metadata.get("chunk_index"),
        }
        formatted.append(f"[chunk-{idx}] {doc.page_content}\nMETA: {meta}")
    return "\n\n".join(formatted)


def _build_graph():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question with the retrieved context. Cite chunk ids like [chunk-0].",
            ),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\nReturn a concise answer.",
            ),
        ]
    )

    qa_chain = prompt | llm | StrOutputParser()

    def retrieve(state: QAState) -> QAState:
        store = _ensure_vector_store()
        k = state.get("k", 4)
        scored = store.similarity_search_with_score(state["query"], k=k)
        documents = [doc for doc, _ in scored]
        scores = [float(score) for _, score in scored]
        return {"documents": documents, "scores": scores}

    def generate(state: QAState) -> QAState:
        context = _format_docs(state.get("documents") or [])
        answer = qa_chain.invoke({"question": state["query"], "context": context})
        return {"answer": answer}

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


qa_graph = _build_graph()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/docs", response_model=UploadResult)
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    documents: List[Document] = []
    for upload in files:
        documents.extend(await _load_file(upload))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        if "page" in chunk.metadata and "page_number" not in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"]

    app.state.vector_store = FAISS.from_documents(chunks, embedding_model)

    return UploadResult(documents=len(documents), chunks=len(chunks))


@app.post("/api/query", response_model=QueryResult)
async def query_documents(payload: QueryPayload):
    _ensure_vector_store()

    state: QAState = {"query": payload.query, "k": payload.k}
    result = qa_graph.invoke(state)

    docs = result.get("documents") or []
    scores = result.get("scores") or []
    sources = []
    for idx, doc in enumerate(docs):
        sources.append(
            {
                "chunk_index": doc.metadata.get("chunk_index", idx),
                "source": doc.metadata.get("source"),
                "page_number": doc.metadata.get("page_number"),
                "score": scores[idx] if idx < len(scores) else None,
            }
        )

    return QueryResult(answer=result.get("answer", ""), sources=sources)

