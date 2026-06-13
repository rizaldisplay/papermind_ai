import numpy as np
from turbovec import IdMapIndex
from app.db.supabase import get_db_connection

class RetrieverService:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        # Inisialisasi Turbovec dengan kuantisasi 4-bit demi menghemat RAM server
        self.index = IdMapIndex(dim=self.dimension, bit_width=4)
        self.chunk_mapping = {}  # Pemetaan internal: incremental_id -> database_chunk_metadata
        self.current_id = 1
        
        self.sync_turbovec_from_pgvector()

    def sync_turbovec_from_pgvector(self):
        """Memuat seluruh data vektor dari pgvector Supabase ke dalam RAM Turbovec saat startup."""
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.book_id, c.content, c.page_number, c.chapter, c.embedding, b.category, b.title
                FROM chunks c
                JOIN books b ON c.book_id = b.id
            """)
            rows = cur.fetchall()
            
            if rows:
                embeddings = []
                ids = []
                for row in rows:
                    # Konversi string pgvector '[0.1, 0.2, ...]' menjadi list float
                    emb_list = [float(x) for x in row['embedding'].strip('[]').split(',')]
                    embeddings.append(emb_list)
                    
                    # Daftarkan ke struktur mapping internal Turbovec
                    inc_id = self.current_id
                    ids.append(inc_id)
                    
                    self.chunk_mapping[inc_id] = {
                        "supabase_id": row['id'],
                        "book_title": row['title'],
                        "category": row['category'],
                        "content": row['content'],
                        "page_number": row['page_number'],
                        "chapter": row['chapter']
                    }
                    self.current_id += 1
                
                # Masukkan tumpukan vektor ke memori kernel Turbovec
                self.index.add_with_ids(
                    np.array(embeddings, dtype=np.float32),
                    np.array(ids, dtype=np.uint64)
                )
        conn.close()

    def add_single_chunk_to_memory(self, embedding: list[float], metadata: dict):
        """Menambahkan data chunk baru secara realtime (Online Ingest) tanpa melatih ulang indeks."""
        inc_id = self.current_id
        emb_np = np.array([embedding], dtype=np.float32)
        id_np = np.array([inc_id], dtype=np.uint64)
        
        self.index.add_with_ids(emb_np, id_np)
        self.chunk_mapping[inc_id] = metadata
        self.current_id += 1

    def search(self, query_vector: list[float], category_filter: str = None, top_k: int = 4) -> list[dict]:
        """Pencarian semantik super cepat menggunakan engine Turbovec."""
        q_v = np.array([query_vector], dtype=np.numpy.float32)
        
        # Bangun filter allowlist jika pengguna memilih kategori buku tertentu
        allowlist = None
        if category_filter:
            allowed_ids = [
                inc_id for inc_id, meta in self.chunk_mapping.items()
                if meta['category'].lower() == category_filter.lower()
            ]
            if allowed_ids:
                allowlist = np.array(allowed_ids, dtype=np.numpy.uint64)
            else:
                return [] # Kategori tidak ditemukan dalam basis pengetahuan

        scores, ids = self.index.search(q_v, k=top_k, allowlist=allowlist)
        
        results = []
        for score, inc_id in zip(scores[0], ids[0]):
            inc_id_int = int(inc_id)
            if inc_id_int in self.chunk_mapping:
                item = self.chunk_mapping[inc_id_int].copy()
                item['score'] = float(score)
                results.append(item)
        return results