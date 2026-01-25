import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

logger = logging.getLogger(__name__)


class MultiStageRetriever:
    """
    Implements a multi-stage retrieval pipeline:
    1. Candidate Generation (BM25 + Vector)
    2. Scoring (Contextual + Temporal)
    3. Re-ranking (Cross-Encoder)
    """

    def __init__(
        self,
        vector_store=None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.vector_store = vector_store
        self.bm25 = None
        self.corpus = []
        self.metadata = []

        self.cross_encoder = None
        if CrossEncoder:
            try:
                # Load lazily or on demand to save memory
                self.cross_encoder_model_name = cross_encoder_model
            except Exception as e:
                logger.warning(f"Failed to initialize CrossEncoder: {e}")

    def update_index(self, documents: List[str], metadatas: List[dict]):
        """Updates the BM25 index with new documents."""
        if not BM25Okapi:
            return

        self.corpus = documents
        self.metadata = metadatas
        tokenized_corpus = [doc.lower().split() for doc in documents]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 index updated with {len(documents)} documents.")

    def _get_temporal_score(
        self, timestamp: Optional[str], lambda_decay: float = 0.1
    ) -> float:
        """Calculates temporal decay score."""
        if not timestamp:
            return 1.0
        try:
            dt = datetime.fromisoformat(timestamp)
            days_old = (datetime.now() - dt).days
            return math.exp(-lambda_decay * days_old)
        except Exception:
            return 1.0

    def retrieve(
        self,
        query: str,
        context: Optional[str] = None,
        limit: int = 5,
        temporal_weight: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Executes the full retrieval pipeline.
        """
        candidates = []
        seen_ids = set()

        # Stage 1: Candidate Generation - BM25
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_n_idx = np.argsort(bm25_scores)[-limit * 2 :]
            for idx in top_n_idx:
                if bm25_scores[idx] > 0:
                    doc = self.corpus[idx]
                    meta = self.metadata[idx]
                    candidates.append(
                        {
                            "content": doc,
                            "metadata": meta,
                            "score": float(bm25_scores[idx]),
                            "source": "bm25",
                        }
                    )
                    if "id" in meta:
                        seen_ids.add(meta["id"])

        # Stage 1: Candidate Generation - Vector
        if self.vector_store:
            vector_results = self.vector_store.query(query, n_results=limit * 2)
            for i in range(len(vector_results.get("documents", [[]])[0])):
                doc = vector_results["documents"][0][i]
                meta = vector_results["metadatas"][0][i]
                doc_id = vector_results["ids"][0][i]

                if doc_id not in seen_ids:
                    candidates.append(
                        {
                            "content": doc,
                            "metadata": meta,
                            "score": 1.0,  # Vector scores need normalization, using 1.0 as base
                            "source": "vector",
                        }
                    )
                    seen_ids.add(doc_id)

        if not candidates:
            return []

        # Stage 2: Scoring (Temporal & Contextual)
        for cand in candidates:
            # Temporal decay
            ts = cand["metadata"].get("timestamp")
            t_score = self._get_temporal_score(ts)
            cand["score"] *= 1.0 - temporal_weight + (temporal_weight * t_score)

            # Contextual relevance (simple keyword match for now)
            if context:
                if context.lower() in cand["content"].lower():
                    cand["score"] *= 1.2

        # Stage 3: Re-ranking (Cross-Encoder)
        if CrossEncoder and len(candidates) > 1:
            if not self.cross_encoder:
                try:
                    self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
                except Exception as e:
                    logger.error(f"Failed to load CrossEncoder for re-ranking: {e}")

            if self.cross_encoder:
                pairs = [[query, cand["content"]] for cand in candidates]
                cross_scores = self.cross_encoder.predict(pairs)
                for i, score in enumerate(cross_scores):
                    candidates[i]["score"] = float(score)
                    candidates[i]["source"] += "+cross-encoder"

        # Sort by final score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:limit]
