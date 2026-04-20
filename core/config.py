import os

class Config:
    """Base configuration class with default settings."""

    COLLECTION_NAME="repay_rag"
    MODEL_NAME="all-MiniLM-L6-v2"
    
    # Chroma settings

    # Logging settings
    LOG_FILE = "app.log"
    DOCS_DIR = "docs"
