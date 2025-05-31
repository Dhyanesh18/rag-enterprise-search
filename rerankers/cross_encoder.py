from sentence_transformers import CrossEncoder
import numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, device="cpu")  # Safer for 4GB VRAM
    
    def rerank(self, query: str, passages: list[dict], top_k: int = 10):
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
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            scores.extend(self.model.predict(batch))
        return scores