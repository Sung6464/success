"""
Configuration management for the LangChain chatbot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # Azure OpenAI - LLM Settings
    AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL", "gpt-4.1")
    AZURE_OPENAI_LLM_API_BASE = os.getenv("AZURE_OPENAI_LLM_MODEL_API_BASE", "")
    AZURE_OPENAI_LLM_API_KEY = os.getenv("AZURE_OPENAI_LLM_MODEL_API_KEY", "")
    AZURE_OPENAI_LLM_API_VERSION = os.getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION", "2024-12-01-preview")
    
    # Azure OpenAI - Embedding Settings
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL", "text-embedding-3-large")
    AZURE_OPENAI_EMBEDDING_API_BASE = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_BASE", "")
    AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_KEY", "")
    AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION", "2024-02-01")
    
    # Legacy OpenAI (for backward compatibility)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    
    # Azure Blob Storage Settings
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_BLOB_STORAGE_CONNECTION_STRING", "")
    AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
    AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")
    AZURE_CONTAINER_NAME = os.getenv("AZURE_BLOB_STORAGE_CONTAINER_NAME", "success-stories")
    
    # PostgreSQL Settings
    POSTGRES_USER = os.getenv("POSTGRESQL_DATABASE_USER", "")
    POSTGRES_PASSWORD = os.getenv("POSTGRESQL_DATABASE_PASSWORD", "")
    POSTGRES_HOST = os.getenv("POSTGRESQL_DATABASE_HOST", "")
    POSTGRES_PORT = os.getenv("POSTGRESQL_DATABASE_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRESQL_DATABASE_DATABASE", "forgex")
    POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_DATABASE_CONNECTION_STRING", "")
    
    # PostgreSQL Connection String
    # @classmethod
    # def get_postgres_connection_string(cls) -> str:
    #     """Generate PostgreSQL connection string."""
    #     return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
    
    # Model Settings (for backward compatibility)
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
    
    # Vector Store Settings
    CHUNK_SIZE = int(os.getenv("VECTOR_STORE_VECTOR_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("VECTOR_STORE_VECTOR_OVERLAP_SIZE", "200"))
    MAX_CHUNKS = int(os.getenv("VECTOR_STORE_MAX_CHUNKS", "500"))
    BATCH_SIZE = int(os.getenv("VECTOR_STORE_BATCH_SIZE", "10"))
    
    # Retriever Settings
    RETRIEVER_K = int(os.getenv("RETRIEVER_RETRIEVER_K", "3"))
    RETRIEVER_SEARCH_TYPE = os.getenv("RETRIEVER_RETRIEVER_SEARCH_TYPE", "similarity_search")
    SIMILARITY_SEARCH_K = int(os.getenv("SIMILARITY_SEARCH_SIM_SEARCH_K", "3"))
    
    # Vector Store Table Name
    VECTOR_STORE_TABLE_NAME = os.getenv("VECTOR_STORE_TABLE_NAME", "success_stories_embeddings")
    VECTOR_STORE_COLLECTION_NAME = os.getenv("VECTOR_STORE_COLLECTION_NAME", "success_stories")

    # SharePoint Settings
    SHAREPOINT_CLIENT_ID = os.getenv("SHAREPOINT_CLIENT_ID")
    SHAREPOINT_CLIENT_SECRET = os.getenv("SHAREPOINT_CLIENT_SECRET")
    SHAREPOINT_TENANT_ID = os.getenv("SHAREPOINT_TENANT_ID")
    SHAREPOINT_SITE_HOSTNAME = os.getenv("SHAREPOINT_SITE_HOSTNAME")
    SHAREPOINT_SITE_PATH = os.getenv("SHAREPOINT_SITE_PATH")
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DOCS_DIR = BASE_DIR / "documents"
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        # Check Azure OpenAI configuration
        if not cls.AZURE_OPENAI_LLM_API_KEY:
            raise ValueError("AZURE_OPENAI_LLM_MODEL_API_KEY is not set in .env file")
        
        if not cls.AZURE_OPENAI_EMBEDDING_API_KEY:
            raise ValueError("AZURE_OPENAI_EMBEDDING_MODEL_API_KEY is not set in .env file")
        
        # Check PostgreSQL configuration
        if not cls.POSTGRES_HOST:
            raise ValueError("POSTGRESQL_DATABASE_HOST is not set in .env file")
        
        if not cls.POSTGRES_USER or not cls.POSTGRES_PASSWORD:
            raise ValueError("PostgreSQL credentials are not set in .env file")
        
        # Check Azure Blob Storage
        if not cls.AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError("AZURE_BLOB_STORAGE_CONNECTION_STRING is not set in .env file")
        
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.DOCS_DIR.mkdir(exist_ok=True)
