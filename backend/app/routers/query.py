from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.embedder import EmbedderService
from app.services.retriever import RetrieverService
from app.services.llm import LLMService
from openai import OpenAI
import os

router = APIRouter(prefix="/query", tags=["Query Pipeline"])

# Inisialisasi Singleton untuk mempertahankan Indeks Turbovec di dalam RAM aplikasi
_retriever_instance = None

def get_retriever():
    global _retriever_instance
    if _retriever_instance == None:
        _retriever_instance = RetrieverService()
    return _retriever_instance

embedder = EmbedderService()
llm = LLMService()

class QueryRequest(BaseModel):
    query: str
    category: str | None = None  # Filter kategori opsional dari pengguna

def rewrite_query(original_query: str) -> str:
    """Komponen Query Rewriting untuk memperbaiki typo dan memperluas sinonim konteks."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a query rewriter. Fix typos and expand synonyms to make it ideal for semantic search. Return ONLY the optimized query text."},
                {"role": "user", "content": original_query}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except:
        return original_query # Fallback ke query asli jika API eksternal terganggu

@router.post("/search-stream")
async def search_and_stream(payload: QueryRequest, retriever: RetrieverService = Depends(get_retriever)):
    # 1. Query Rewriting Layer
    optimized_query = rewrite_query(payload.query)
    
    # 2. Pembuatan Representasi Vektor Query
    query_vector = embedder.get_embedding(optimized_query)
    
    # 3. Pencarian Semantik Menggunakan Mesin Turbovec
    matched_chunks = retriever.search(query_vector, category_filter=payload.category, top_k=3)
    
    # 4. Confidence Scoring & Routing Layer
    # Ambil skor tertinggi dari daftar kecocokan terdekat
    highest_score = matched_chunks[0]['score'] if matched_chunks else 0.0
    
    # Sesuai diagram alur: Skor >= 0.75 -> RAG Answer, < 0.75 -> LLM Fallback
    if highest_score >= 0.75:
        return StreamingResponse(
            llm.generate_rag_response(payload.query, matched_chunks),
            media_type="text/event-stream"
        )
    else:
        return StreamingResponse(
            llm.generate_fallback_response(payload.query),
            media_type="text/event-stream"
        )