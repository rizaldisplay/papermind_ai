import fitz  # PyMuPDF
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from app.db.supabase import get_db_connection
from app.services.chunker import ChunkerService
from app.services.embedder import EmbedderService
from app.routers.query import get_retriever

router = APIRouter(prefix="/ingest", tags=["Ingestion Pipeline"])
embedder = EmbedderService()

@router.post("/book")
async def ingest_pdf_book(
    title: str = Form(..., description="Judul buku atau dokumen"),
    author: str = Form(None, description="Nama penulis (opsional)"),
    category: str = Form(..., description="Kategori utama untuk filtering/routing"),
    file: UploadFile = File(..., description="Berkas PDF yang akan diekstrak"),
    retriever = Depends(get_retriever)
):
    # 1. Validasi format file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Format berkas harus berupa .pdf")
        
    conn = get_db_connection()
    try:
        # 2. Baca file PDF ke dalam memori stream bytes
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        if total_pages == 0:
            raise HTTPException(status_code=400, detail="Berkas PDF kosong atau rusak.")

        # 3. Ekstrak teks halaman demi halaman dan beri pembatas [PAGE_BREAK]
        full_text_with_markers = ""
        for page_num in range(total_pages):
            page = doc[page_num]
            # Ambil teks bersih dari halaman
            page_text = page.get_text("text")
            full_text_with_markers += page_text + f"\n[PAGE_BREAK]\n"
        
        doc.close() # Tutup dokumen setelah ekstraksi selesai

        with conn.cursor() as cur:
            # 4. Simpan Metadata Utama ke Tabel Books di Supabase
            # file_path diisi dengan nama file asli sebagai penanda
            cur.execute("""
                INSERT INTO books (title, author, category, file_path, total_pages)
                VALUES (%s, %s, %s, %s, %s) RETURNING id;
            """, (title, author, category, file.filename, total_pages))
            book_id = cur.fetchone()['id']
            
            # 5. Segmentasi Teks menggunakan ChunkerService yang sudah kita buat sebelumnya
            chunks = ChunkerService.split_text(full_text_with_markers)
            
            for chunk in chunks:
                # 6. Hitung Vector Embedding (1536 Dimensi via OpenAI)
                embedding = embedder.get_embedding(chunk['content'])
                
                # 7. Amankan secara Permanen ke PostgreSQL (pgvector)
                cur.execute("""
                    INSERT INTO chunks (book_id, content, page_number, chapter, embedding)
                    VALUES (%s, %s, %s, %s, %s) RETURNING id;
                """, (book_id, chunk['content'], chunk['page_number'], chunk['chapter'], embedding))
                chunk_uuid = cur.fetchone()['id']
                
                # 8. Akselerasikan langsung ke In-Memory Index Turbovec secara Real-time
                retriever.add_single_chunk_to_memory(
                    embedding=embedding,
                    metadata={
                        "supabase_id": chunk_uuid,
                        "book_title": title,
                        "category": category,
                        "content": chunk['content'],
                        "page_number": chunk['page_number'],
                        "chapter": chunk['chapter']
                    }
                )
            conn.commit()
            
        return {
            "status": "success", 
            "message": f"Berhasil memproses dokumen '{title}'.",
            "details": {
                "filename": file.filename,
                "total_pages": total_pages,
                "total_chunks_created": len(chunks)
            }
        }
        
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Gagal memproses PDF: {str(e)}")
    finally:
        conn.close()