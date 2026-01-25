import logging
import os
from typing import Any, List, Optional, Dict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class CachedEmbeddingFunction(EmbeddingFunction[Documents]):
    """Wrapper for embedding functions that adds caching."""

    def __init__(self, ef: Any, cache_manager: Optional[CacheManager] = None):
        self.ef = ef
        self.cache_manager = cache_manager

    def __call__(self, input: Documents) -> Embeddings:
        if not self.cache_manager:
            return self.ef(input)

        results = [None] * len(input)
        to_embed = []
        indices_to_embed = []

        # Check cache first
        for i, text in enumerate(input):
            cached = self.cache_manager.get_embedding(text)
            if cached is not None:
                results[i] = cached
            else:
                to_embed.append(text)
                indices_to_embed.append(i)

        # Embed missing items
        if to_embed:
            new_embeddings = self.ef(to_embed)
            for i, emb in enumerate(new_embeddings):
                # Ensure we store as list if it's a numpy array
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                self.cache_manager.set_embedding(to_embed[i], emb)
                results[indices_to_embed[i]] = emb

        return results

    @classmethod
    def name(cls) -> str:
        """Returns the name of the embedding function.
        Using 'default' to match common persisted configurations and avoid conflicts.
        """
        return "default"

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration for the embedding function."""
        return {}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "CachedEmbeddingFunction":
        """Builds a CachedEmbeddingFunction from a configuration."""
        return cls(SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"))

    def embed_query(self, input: Any) -> Any:
        """Handle query embedding, supporting both single string and list of strings."""
        if isinstance(input, str):
            if self.cache_manager:
                cached = self.cache_manager.get_embedding(input)
                if cached is not None:
                    return cached

            if hasattr(self.ef, "embed_query"):
                res = self.ef.embed_query(input)
            else:
                res = self.ef([input])[0]

            if hasattr(res, "tolist"):
                res = res.tolist()

            if self.cache_manager:
                self.cache_manager.set_embedding(input, res)
            return res
        elif isinstance(input, list):
            return self.__call__(input)
        return self.ef(input)


class VectorStore:
    """
    Handles persistent vector storage and semantic search using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str = ".chroma_db",
        cache_manager: Optional[CacheManager] = None,
    ):
        self.persist_directory = persist_directory
        self.cache_manager = cache_manager
        self.client = chromadb.PersistentClient(
            path=self.persist_directory, settings=Settings(allow_reset=True)
        )

        # Using explicit embedding function (SentenceTransformer)
        base_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.embedding_function = CachedEmbeddingFunction(base_ef, cache_manager)

        self.collection = self.client.get_or_create_collection(
            name="project_knowledge", embedding_function=self.embedding_function
        )

    def add_documents(
        self, documents: List[str], metadatas: List[dict], ids: List[str]
    ):
        """Adds documents to the vector store."""
        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(documents)} documents to ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")

    def query(self, query_text: str, n_results: int = 5) -> dict:
        """Performs a semantic search."""
        try:
            results = self.collection.query(
                query_texts=[query_text], n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"documents": [], "metadatas": [], "ids": []}

    def delete_collection(self):
        """Deletes the current collection."""
        self.client.delete_collection("project_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="project_knowledge", embedding_function=self.embedding_function
        )

    def get_count(self) -> int:
        """Returns the number of items in the collection."""
        return self.collection.count()
