"""
RAG (Retrieval-Augmented Generation) pipeline for Hindi NLP tasks.

Uses FAISS vector store with multilingual embeddings to retrieve
relevant Hindi documents and augment model prompts with context.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class HindiRAGPipeline:
    """
    RAG pipeline for Hindi NLP tasks.

    Components:
    1. Document embeddings using multilingual sentence transformers
    2. FAISS vector store for efficient similarity search
    3. Context-augmented prompt construction
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embedder = None
        self.index = None
        self.documents = []

    def build_index(self, documents: List[str]) -> None:
        """
        Build FAISS index from documents.

        Args:
            documents: List of document texts to index
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            logger.info(f"Building RAG index with {len(documents)} documents...")

            # Chunk documents
            chunks = []
            for doc in documents:
                doc_chunks = self._chunk_text(doc)
                chunks.extend(doc_chunks)

            self.documents = chunks
            logger.info(f"  Created {len(chunks)} chunks")

            # Embed chunks
            if self.embedder is None:
                self.embedder = SentenceTransformer(self.embedding_model_name)

            embeddings = self.embedder.encode(chunks, show_progress_bar=True, batch_size=32)
            embeddings = np.array(embeddings, dtype=np.float32)

            # Build FAISS index
            import faiss
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity after normalization)

            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)

            logger.success(f"RAG index built: {self.index.ntotal} vectors, {dim} dimensions")

        except ImportError as e:
            logger.warning(f"RAG dependencies not installed: {e}")
            logger.info("Falling back to simple keyword matching")
            self.documents = documents
            self.index = None

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of relevant document chunks
        """
        k = top_k or self.top_k

        if self.index is not None and self.embedder is not None:
            import numpy as np
            import faiss

            query_embedding = self.embedder.encode([query])
            query_embedding = np.array(query_embedding, dtype=np.float32)
            faiss.normalize_L2(query_embedding)

            distances, indices = self.index.search(query_embedding, k)

            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.documents):
                    results.append(self.documents[idx])

            return results
        else:
            # Fallback: keyword matching
            query_words = set(query.lower().split())
            scored = []
            for doc in self.documents:
                doc_words = set(doc.lower().split())
                overlap = len(query_words & doc_words)
                scored.append((overlap, doc))

            scored.sort(reverse=True)
            return [doc for _, doc in scored[:k]]

    def augment_prompt(
        self,
        original_prompt: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Augment a prompt with retrieved context.

        Args:
            original_prompt: The original task prompt
            query: The search query (usually the input text)
            top_k: Number of context documents

        Returns:
            Augmented prompt with context
        """
        retrieved = self.retrieve(query, top_k)

        if not retrieved:
            return original_prompt

        context = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(retrieved)])

        augmented = (
            f"Use the following context to help with the task:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"{original_prompt}"
        )

        return augmented

    def save_index(self, path: str) -> None:
        """Save the FAISS index and documents to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            import faiss
            faiss.write_index(self.index, str(path / "faiss.index"))

        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

        logger.info(f"RAG index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load the FAISS index and documents from disk."""
        path = Path(path)

        if (path / "faiss.index").exists():
            import faiss
            self.index = faiss.read_index(str(path / "faiss.index"))

        if (path / "documents.json").exists():
            with open(path / "documents.json", "r", encoding="utf-8") as f:
                self.documents = json.load(f)

        logger.info(f"RAG index loaded from {path}")
