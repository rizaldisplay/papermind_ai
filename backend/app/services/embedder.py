import os
import cohere
from dotenv import load_dotenv

load_dotenv()

class EmbedderService:
    def __init__(self):
        # JALUR API: Mendefinisikan kembali self.co
        self.co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        self.model = "embed-multilingual-v3.0" 

    def get_embedding(self, text: str, input_type: str = "search_document") -> list[float]:
        text = text.replace("\n", " ")
        
        # Memanggil API Cohere
        response = self.co.embed(
            texts=[text],
            model=self.model,
            input_type=input_type,
            embedding_types=["float"]
        )
        return response.embeddings.float[0]