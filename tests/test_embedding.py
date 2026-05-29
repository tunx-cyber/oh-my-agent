from core.embedding import EmbeddingClient
from config.settings import get_settings

s = get_settings()
emb = EmbeddingClient(
    api_key=s.EMBEDDING_API_KEY,
    api_url=s.EMBEDDING_API_URL,
    model=s.EMBEDDING_MODEL,
)

print(emb.embed("hello")[:5])
