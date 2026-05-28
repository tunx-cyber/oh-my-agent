class BaseEmbeddingAdapter:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
    
    def get_embeddings(self,text):
        pass