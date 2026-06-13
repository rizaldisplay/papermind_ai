from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.embedder import EmbedderService
from app.services.retriever import RetrieverService
from app.services.llm import LLMService

router = APIRouter(prefix="/query", tags=["Query Pipeline"])

# Inisialisasi Singleton untuk mempertahankan Indeks Turbovec di dalam RAM aplikasi
_retriever_instance = None

def get_retriever() -> RetrieverService:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RetrieverService()
    return _retriever_instance

# Registrasi Services sesuai struktur backend/app/services/
embedder = EmbedderService()
llm = LLMService()

class QueryRequest(BaseModel):
    query: str
    category: str | None = None  # Filter utama berdasarkan kolom kategori di tabel 'books'

@router.post("/search-stream")
async def search_and_stream(
    payload: QueryRequest, 
    retriever: RetrieverService = Depends(get_retriever)
):
    # LALUAN 1: Query Rewriting Layer
    # Mendelegasikan perbaikan typo & ekspansi sinonim ke LLM Service secara asynchronous
    optimized_query = await llm.rewrite_query(payload.query)
    
    # LALUAN 2: Pembuatan Representasi Vektor Query
    # Mengonversi teks query hasil optimasi menjadi dense vector (1536 / 1024 dim)
    query_vector = embedder.get_embedding(optimized_query, input_type="search_query") 
   
    # LALUAN 3: Semantic Search Engine
    # Menembak mesin pencari dengan filter kategori opsional untuk mengambil top-K chunks
    matched_chunks = retriever.search(
        query_vector, 
        category_filter=payload.category, 
        top_k=3
    )
    
    # LALUAN 4: Confidence Scoring & Routing Layer
    # Memeriksa skor kecocokan tertinggi (Cosine Similarity) dari data teratas
    highest_score = matched_chunks[0]['score'] if matched_chunks else 0.0
    
    # Penentuan rute respons berdasarkan Score Threshold di dokumen arsitektur:
    # Skor >= 0.75 -> RAG Answer (Streaming teks + sitasi data sumber)
    if highest_score >= 0.75:
        return StreamingResponse(
            llm.generate_rag_response(payload.query, matched_chunks),
            media_type="text/event-stream"
        )
    
    # Skor < 0.75 -> LLM Fallback (General knowledge dengan disclaimer "[not in library]")
    else:
        return StreamingResponse(
            llm.generate_fallback_response(payload.query),
            media_type="text/event-stream"
        )