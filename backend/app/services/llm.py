import os
import json
import cohere
import asyncio
# from openai import OpenAI
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        self.co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        # Command-r sangat dioptimasi untuk akurasi pengerjaan skenario RAG
        self.model = "command-a-03-2025"
    
    async def rewrite_query(self, original_query: str) -> str:
        """
        Mengoptimalkan query pengguna sebelum masuk ke pencarian semantik.
        """
        def _execute_sync_rewrite():
            try:
                response = self.co.chat(
                    model=self.model,
                    message=original_query,
                    preamble=(
                        "You are a query rewriter. Fix typos and expand synonyms "
                        "to make it ideal for semantic search. Return ONLY the optimized query text."
                    )
                )
                return response.text.strip()
            except Exception as e:
                print(f"[LLM-SERVICE-ERROR] Gagal rewrite query: {e}")
                return original_query

        # Menjalankan fungsi blocking I/O di dalam worker thread terpisah
        return await asyncio.to_thread(_execute_sync_rewrite)

    async def generate_rag_response(self, query: str, contexts: list[dict]) -> AsyncGenerator[str, None]:
        context_str = "\n\n".join([
            f"Source: {c['book_title']} | Chapter: {c['chapter']} | Page: {c['page_number']}\nContent: {c['content']}"
            for c in contexts
        ])
        
        preamble = (
            "You are a professional Perplexity-like research assistant. Answer the user's question based strictly on the provided context.\n"
            "At the end of your complete answer, you MUST provide a separate citations block listing the book title, chapter, and page number."
        )
        
        message = f"Context:\n{context_str}\n\nQuestion: {query}"
        
        response = self.co.chat_stream(
            model=self.model,
            message=message,
            preamble=preamble
        )
        
        citations = [{
            "title": c['book_title'], "chapter": c['chapter'], "page": c['page_number']
        } for c in contexts]
        yield f"__METADATA__:{json.dumps({'status': 'RAG_FOUND', 'citations': citations})}\n"

        for chunk in response:
            # Event type 'text-generation' menandakan kedatangan token teks baru
            if chunk.event_type == "text-generation":
                yield chunk.text

    async def generate_fallback_response(self, query: str) -> AsyncGenerator[str, None]:
        preamble = (
            "You are a helpful AI assistant. The requested information was not found in our local document library.\n"
            "Answer the question using your general knowledge, but you MUST start your response with the explicit label '[NOT IN LIBRARY]' on a new line."
        )
        
        response = self.co.chat_stream(
            model=self.model,
            message=query,
            preamble=preamble
        )
        
        yield f"__METADATA__:{json.dumps({'status': 'FALLBACK_LLM', 'citations': []})}\n"
        
        for chunk in response:
            if chunk.event_type == "text-generation":
                yield chunk.text