"""
Configuration Module for Supply Chain LLMOps Application.

This module centralizes all configuration settings including:
- Azure OpenAI API credentials and endpoints
- Model deployment configurations
- Vector store settings (Chroma/Azure Search)
- Logging setup

All settings can be overridden via environment variables defined in .env file.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure application-wide logging
logger = logging.getLogger("supply_chain_llmops")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
logger.addHandler(handler)


class Config:
    """
    Central configuration class for application settings.

    This class loads and validates all environment variables required for the
    application to function. It provides default values where appropriate and
    raises errors for missing critical configuration.

    Attributes:
        AZURE_OPENAI_API_KEY (str): Azure OpenAI API key
        AZURE_OPENAI_ENDPOINT (str): Azure OpenAI endpoint URL
        AZURE_OPENAI_API_VERSION (str): API version for Azure OpenAI
        AZURE_DEPLOYMENT (str): Deployment name for main LLM (e.g., gpt-5-mini)
        AZURE_JUDGE_DEPLOYMENT (str): Deployment name for LLM Judge (e.g., gpt-4.1)
        AZURE_EMBEDDING_DEPLOYMENT (str): Deployment name for embeddings
        VECTOR_STORE_TYPE (str): Type of vector store ('chroma' or 'azure')
        AZURE_SEARCH_ENDPOINT (str): Azure AI Search endpoint (if using Azure)
        AZURE_SEARCH_KEY (str): Azure AI Search API key (if using Azure)
        AZURE_SEARCH_INDEX (str): Azure AI Search index name (if using Azure)
    """

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
    AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    AZURE_JUDGE_DEPLOYMENT = os.getenv("AZURE_JUDGE_DEPLOYMENT", "gpt-4.1")

    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")  # 'chroma' or 'azure'
    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "supply-chain-index")

    @staticmethod
    def validate():
        """
        Validate that required configuration is present.

        Logs a warning if neither OpenAI nor Azure OpenAI API keys are configured.
        This is a soft validation - the application may still function with certain
        limitations without these keys.
        """
        if not Config.OPENAI_API_KEY and not Config.AZURE_OPENAI_API_KEY:
            logger.warning(
                "No OpenAI API Key found. Please set OPENAI_API_KEY or "
                "AZURE_OPENAI_API_KEY in .env"
            )


# Initialize tracer placeholder (App Insights removed, kept for compatibility)
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
