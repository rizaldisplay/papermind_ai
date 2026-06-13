import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

try:
    conn = psycopg2.connect(
        DATABASE_URL
    )
    print("CONNECTED")
except Exception as e:
    print(e)

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

def init_db():
    """Inisialisasi ekstensi pgvector dan tabel jika belum ada."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Tabel Books
        cur.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title TEXT NOT NULL,
                author TEXT,
                category TEXT,
                file_path TEXT,
                total_pages INT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Tabel Chunks
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                book_id UUID REFERENCES books(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                page_number INT,
                chapter TEXT,
                embedding vector(1024),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        conn.commit()
    conn.close()