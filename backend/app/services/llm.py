import os
import json
import cohere
import asyncio
# from openai import OpenAI
from typing import AsyncGenerator
from dotenv import load_dotenv
from .gemini import GeminiService
from .web_search_firecrawl import FirecrawlWebSearch
from .web_search_exa import ExaWebSearch

load_dotenv()

class LLMService:
    def __init__(self):
        self.co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        # Command-r sangat dioptimasi untuk akurasi pengerjaan skenario RAG
        self.model = "command-a-03-2025"
        
        # self.gemini = GeminiService()
        if env("EXA_API_KEY", default=None):
            self.search_tool = ExaWebSearch()
        else:
            self.search_tool = FirecrawlWebSearch()
    
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

    async def generate_search_queries(self, messages: List[Dict[str, Any]]) -> List[str]:
                """Generate 3 different web search queries based on user message using LLM"""
                last_message = self._get_last_user_message(messages)
                if not last_message:
                    return []
                
                # Get conversation context (last 3-5 messages)
                conversation_context = self._get_conversation_context(messages, max_messages=5)
                
                # Get current date information
                current_date = datetime.now()
                date_context = f"""Current date: {current_date.strftime('%B %d, %Y')}
        Current year: {current_date.year}
        Current month: {current_date.strftime('%B')}"""
                
                query_messages = [
                    {
                        "role": "user",
                        "content": f"""Generate 3 different web search queries based on the user's latest message within the context of the conversation. 
        The queries should approach the topic from different angles or search for different aspects.

        {date_context}

        Conversation context:
        {conversation_context}

        Latest user message: {last_message}

        Important guidelines:
        - Consider the conversation context to understand what the user is really asking about
        - If the latest message refers to "it", "this", "that", etc., use the context to understand what is being referenced
        - If the user asks about "latest", "recent", "current", or "new" information, include the current year ({current_date.year}) or month in relevant queries
        - For time-sensitive topics, add appropriate year or date qualifiers
        - For comparisons or updates, consider including the current year
        - Only add date/year when it makes the search more relevant

        Return ONLY the 3 search queries, one per line, no explanation or numbering:"""
                    }
                ]
        
                try:
                    response = await self.gemini.client.messages.create(
                        model="claude-3-5-haiku-latest",
                        max_tokens=150,
                        messages=query_messages
                    )
                    
                    queries_text = response.content[0].text.strip()
                    # Split by newlines and clean up
                    queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
                    
                    # Basic validation and limit to 3
                    valid_queries = []
                    for query in queries[:3]:
                        if 5 < len(query) < 100:
                            valid_queries.append(query)
                    
                    # Log generated search queries
                    print("\n" + "="*60)
                    print("🔍 GENERATED SEARCH QUERIES:")
                    for i, query in enumerate(valid_queries, 1):
                        print(f"  {i}. {query}")
                    print("="*60 + "\n")
                    
                    return valid_queries
                        
                except Exception as e:
                    print(f"Query generation failed: {e}")
                
                return []
    
