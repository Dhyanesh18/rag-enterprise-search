"""
Cross-Encoder Reranker Module

This module defines a CrossEncoderReranker class that uses a sentence-transformers
CrossEncoder model (e.g., ms-marco-MiniLM-L-6-v2) to rerank retrieved passages
based on their semantic relevance to a query.

It is designed for use in RAG or semantic search pipelines where initial retrieval
may be based on embeddings or sparse methods (e.g., BM25), and a more accurate 
re-ranking step is needed using a cross-encoder.
"""
from sentence_transformers import CrossEncoder
import numpy as np

class CrossEncoderReranker:
    """
    Uses a sentence-transformers CrossEncoder model to rerank passages
    based on their semantic relevance to a given query.
    """
    def __init__(self, model_path="./models/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the cross-encoder reranker.

        Args:
            model_path (str): Local path or model name from HuggingFace Hub.
                              Recommended model: ms-marco-MiniLM-L-6-v2.
        """
        self.model = CrossEncoder(model_path, device="cpu")  # Safer for 4GB VRAM
    
    def rerank(self, query: str, passages: list[dict], top_k: int = 10):
        """
        Reranks a list of retrieved passages using the cross-encoder model.

        Args:
            query (str): User query.
            passages (list[dict]): List of retrieved chunks/passages. Each dict must contain at least a "text" field.
            top_k (int): Number of top passages to return after reranking.

        Returns:
            list[dict]: Top-k passages sorted by cross-encoder score. Each passage includes:
                        - "rerank_score": score from the cross-encoder
                        - "original_score": score from the first retrieval stage
                        - "score": updated to rerank_score for downstream compatibility
        """
        if not passages:
            return []
    
        # Keep original scores for reference
        original_scores = [p.get("score", 0) for p in passages]
    
        # Batch prediction
        pairs = [(query, p["text"]) for p in passages]
        scores = self._predict_in_batches(pairs, batch_size=8)
    
        # Merge scores with original context
        scored_passages = []
        for p, new_score in zip(passages, scores):
            p["rerank_score"] = new_score
            p["original_score"] = p.get("score", 0)
            p["score"] = new_score  # Use cross-encoder as primary score
            scored_passages.append(p)
    
        return sorted(scored_passages, key=lambda x: x["score"], reverse=True)[:top_k]
    
    def _predict_in_batches(self, pairs, batch_size=8):
        """
        Performs batched inference with the cross-encoder model.

        Args:
            pairs (list[tuple]): List of (query, passage) pairs.
            batch_size (int): Number of pairs to process per batch.

        Returns:
            list[float]: Scores from the model for each pair.
        """
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            scores.extend(self.model.predict(batch))
        return scores
