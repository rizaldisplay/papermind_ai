import json
import environ
from google import genai
from google.genai import types

env = environ.Env(
    GEMINI_API_KEY=str,
)

class GeminiService:
    def __init__(self):
        # Inisialisasi client Gemini menggunakan API Key dari .env
        self.client = genai.Client(api_key=env("GEMINI_API_KEY"))

    async def stream_response(self, messages, model="gemini-2.5-flash", enable_web_search=False):
        # 1. ADAPTASI FORMAT PESAN (Drop-in Replacement)
        # Claude/OpenAI menggunakan role: 'assistant', sedangkan Gemini menggunakan 'model'
        formatted_contents = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"

            formatted_contents.append({
                "role": role,
                "parts": [msg.get("content", "")]
            })

        # 2. KONFIGURASI PARAMETER & FITUR
        config_params = {
            "max_output_tokens": 1024,
        }

        # Jika Web Search di-true-kan dari Frontend, Google Grounding langsung aktif
        if enable_web_search:
            config_params["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        config = types.GenerateContentConfig(**config_params)

        try:
            # Menggunakan namespace .aio untuk mode Asynchronous
            response_stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=formatted_contents,
                config=config
            )

            async for chunk in response_stream:
                if chunk.text:
                    # Stream teks menggunakan standar Vercel AI SDK
                    yield f'0:{json.dumps(chunk.text)}\n'

            # Kirim sinyal d: (done) saat generate selesai
            yield f'd:{json.dumps({"finishReason": "stop"})}\n'

        except Exception as e:
            # Kirim sinyal 3: (error) jika terjadi kegagalan
            yield f'3:{json.dumps({"error": str(e)})}\n'