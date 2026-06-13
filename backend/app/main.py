from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.supabase import init_db
from app.routers import ingest, query

app = FastAPI(
    title="Perplexity Clone Core RAG Engine",
    description="FastAPI Backend with pgvector Persistence and Turbovec CPU Acceleration",
    version="1.0.0"
)

# Konfigurasi CORS agar frontend dapat mengakses API secara aman
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    print("[SYSTEM] Initializing Supabase Table Schema...")
    init_db()
    
    # Memicu pembentukan singleton instansiasi indeks Turbovec sejak pertama kali dinyalakan
    print("[SYSTEM] Pre-loading pgvector records into memory space (Turbovec)...")
    query.get_retriever()
    print("[SYSTEM] Hybrid Core Engine RAG Ready for Production.")

@app.get("/")
def read_root():
    return {"status": "online", "engine": "FastAPI + pgvector + Turbovec"}

# Registrasi Router Komponen
app.include_router(ingest.router)
app.include_router(query.router)