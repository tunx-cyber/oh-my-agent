from openai import OpenAI
from .base_embed_adapter import BaseEmbeddingAdapter
class OpenaiEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str
    ):
        super().__init__(api_url=api_url, api_key=api_key,model=model)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )
    
    def get_embeddings(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding