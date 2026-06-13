import os
import json
from openai import OpenAI
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_rag_response(self, query: str, contexts: list[dict]) -> AsyncGenerator[str, None]:
        """Membuat jawaban berbasis dokumen RAG (Kondisi: Found)."""
        context_str = "\n\n".join([
            f"Source: {c['book_title']} | Chapter: {c['chapter']} | Page: {c['page_number']}\nContent: {c['content']}"
            for c in contexts
        ])
        
        system_prompt = (
            "You are a professional Perplexity-like research assistant. Answer the user's question based strictly on the provided context.\n"
            "At the end of your complete answer, you MUST provide a separate citations block listing the book title, chapter, and page number."
        )
        
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )
        
        # Kirim metadata sitasi terlebih dahulu sebagai token inisial agar frontend bisa merendernya
        citations = [{
            "title": c['book_title'], "chapter": c['chapter'], "page": c['page_number']
        } for c in contexts]
        yield f"__METADATA__:{json.dumps({'status': 'RAG_FOUND', 'citations': citations})}\n"

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    async def generate_fallback_response(self, query: str) -> AsyncGenerator[str, None]:
        """Membuat jawaban berbasis pengetahuan umum jika skor similarity rendah (Kondisi: Fallback)."""
        system_prompt = (
            "You are a helpful AI assistant. The requested information was not found in our local document library.\n"
            "Answer the question using your general knowledge, but you MUST start your response with the explicit label '[NOT IN LIBRARY]' on a new line."
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            stream=True
        )
        
        yield f"__METADATA__:{json.dumps({'status': 'FALLBACK_LLM', 'citations': []})}\n"
        
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token