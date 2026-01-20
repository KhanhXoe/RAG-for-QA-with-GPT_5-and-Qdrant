from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field

class Settings(BaseSettings):
    openai_api_key: str = Field(...)
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = Field(default="documents")
    
    router_model: str = "gpt-5-2025-08-07"
    reformulator_model: str = "gpt-5-2025-08-07"
    completion_model: str = "gpt-5-2025-08-07"
    answer_model: str = "gpt-5-2025-08-07"
    embedding_model: str = "text-embedding-3-large"
    completion_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    @field_validator('completion_threshold', mode='before')
    @classmethod
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('completion_threshold must be between 0.0 and 1.0')
        return v
    
    @field_validator('openai_api_key', mode='before')
    @classmethod
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError('openai_api_key cannot be empty')
        return v.strip()
    
    class Config:
        env_file = ".env" 