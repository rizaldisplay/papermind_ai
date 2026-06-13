import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class EmbedderService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small" # Menghasilkan dimensi 1536

    def get_embedding(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding