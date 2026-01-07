import os
import json
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from helper import create_chunks_with_overlap
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import requests

load_dotenv()

def _ensure_hf_key():
    key = os.getenv("HF_KEY")
    if not key:
        raise RuntimeError("HF_KEY is not set in environment or .env")
    if not key.lower().startswith("hf_"):
        raise RuntimeError("HF_KEY does not look like a valid HuggingFace key (expected prefix hf_)")
    return key

# Initialize embeddings with key check
hf_key = _ensure_hf_key()
embeddings = HuggingFaceEndpointEmbeddings(
    model="intfloat/multilingual-e5-large-instruct",
    huggingfacehub_api_token=hf_key,
    # task="sentence-similarity"
)

def test_embeddings_creation():
    """Test creating embeddings with FAISS and HuggingFace"""
    # Smoke test for embed_query
    print("Running HF embed_query smoke test...")
    print(f"Using HF_KEY: {hf_key[:8]}...{hf_key[-8:]}")
    
    smoke = embeddings.embed_query("hello world")
    print(f"✓ HF embed_query smoke test, dim={len(smoke)}")


    
    # Sample test data
    test_texts = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "pageNum": 1
        },
        {
            "text": "Deep learning uses neural networks with multiple layers to process data.",
            "pageNum": 2
        },
        {
            "text": "Natural language processing helps computers understand and generate human language.",
            "pageNum": 3
        }
    ]
    
    # Create chunks with overlap
    all_chunks = []
    for item in test_texts:
        chunks = create_chunks_with_overlap(item["text"], chunk_size=50, overlap_size=10, page_number=item["pageNum"])
        all_chunks.extend(chunks)
    
    # Create documents
    documents = []
    for chunk in all_chunks:
        documents.append(Document(
            page_content=chunk['chunk'],
            metadata=chunk['metadata']
        ))
    
    print(f"✓ Created {len(documents)} document chunks")
    
    # Create FAISS vector store
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vectorstore = FAISS.from_documents(documents=documents, ids=uuids, embedding=embeddings)
    print("✓ FAISS vector store created successfully")
    
    # Save vector store
    vectorstore.save_local("test_vector_store")
    print("✓ Vector store saved locally")
    
    # Load and test retrieval
    loaded_vectorstore = FAISS.load_local(
        "test_vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✓ Vector store loaded successfully")
    
    # Test similarity search
    query = "neural networks"
    results = loaded_vectorstore.similarity_search_with_relevance_scores(query, k=2)
    print(f"✓ Similarity search results for '{query}':")
    for doc, score in results:
        print(f"  - Score: {score:.4f}, Content: {doc.page_content[:50]}...")
    
    print("\n✓ All embedding tests passed!")

if __name__ == "__main__":
    test_embeddings_creation()