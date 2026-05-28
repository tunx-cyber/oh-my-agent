from core.embedding_adapters.openai_adapter import OpenaiEmbeddingAdapter
from config.settings import get_settings
s = get_settings()
emb = OpenaiEmbeddingAdapter(
    api_key=s.EMBEDDING_API_KEY,
    api_url=s.EMBEDDING_API_URL,
    model=s.EMBEDDING_MODEL
)

print(emb.get_embeddings("hello")[:5])