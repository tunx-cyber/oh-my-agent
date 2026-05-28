from pydantic_settings import BaseSettings,SettingsConfigDict
from functools import lru_cache
class Settings(BaseSettings):
    API_URL: str
    API_KEY: str
    MODEL: str

    TEST_API_URL: str
    TEST_API_KEY: str
    TEST_MODEL: str

    EMBEDDING_API_URL: str
    EMBEDDING_API_KEY: str
    EMBEDDING_MODEL: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache()
def get_settings() -> Settings:
    s = Settings()
    return s