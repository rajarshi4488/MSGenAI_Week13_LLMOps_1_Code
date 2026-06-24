"""
Vector Store Abstraction Layer.

This module provides a unified interface for vector storage and retrieval,
supporting both local (ChromaDB) and cloud (Azure AI Search) backends.

The abstraction allows switching between vector stores via configuration
without changing application code. Both implementations support:
- Adding documents with automatic embedding generation
- Similarity search for retrieval augmented generation (RAG)

Classes:
    VectorStoreInterface: Abstract base class defining the contract
    ChromaVectorStore: Local ChromaDB implementation
    AzureVectorStore: Cloud Azure AI Search implementation

Functions:
    get_embeddings: Factory for Azure/OpenAI embedding models
    get_vector_store: Factory returning the configured vector store
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Any

# LangChain imports
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, AzureSearch
from src.config import Config

logger = logging.getLogger("supply_chain_llmops")


def get_embeddings():
    """
    Create and return configured embedding model.

    Uses Azure OpenAI embeddings if configured, falls back to OpenAI.

    Returns:
        AzureOpenAIEmbeddings | OpenAIEmbeddings: Configured embedding model
    """
    if Config.AZURE_OPENAI_API_KEY:
        return AzureOpenAIEmbeddings(
            azure_deployment=Config.AZURE_EMBEDDING_DEPLOYMENT,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )
    else:
        return OpenAIEmbeddings()


class VectorStoreInterface(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        pass


class ChromaVectorStore(VectorStoreInterface):
    def __init__(self):
        self.embeddings = get_embeddings()
        self.persist_directory = os.path.join(os.getcwd(), "chroma_db")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="supply_chain_knowledge",
        )

    def add_documents(self, documents: List[Document]):
        logger.info(f"Adding {len(documents)} documents to ChromaDB.")
        self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)


class AzureVectorStore(VectorStoreInterface):
    def __init__(self):
        self.embeddings = get_embeddings()
        if not Config.AZURE_SEARCH_ENDPOINT or not Config.AZURE_SEARCH_KEY:
            raise ValueError("Azure Search Endpoint and Key are required for AzureVectorStore.")

        self.vector_store = AzureSearch(
            azure_search_endpoint=Config.AZURE_SEARCH_ENDPOINT,
            azure_search_key=Config.AZURE_SEARCH_KEY,
            index_name=Config.AZURE_SEARCH_INDEX,
            embedding_function=self.embeddings,
        )

    def add_documents(self, documents: List[Document]):
        logger.info(f"Adding {len(documents)} documents to Azure AI Search.")
        self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)


def get_vector_store() -> VectorStoreInterface:
    store_type = Config.VECTOR_STORE_TYPE.lower()
    logger.info(f"Initializing Vector Store: {store_type}")

    if store_type == "azure":
        return AzureVectorStore()
    else:
        return ChromaVectorStore()
