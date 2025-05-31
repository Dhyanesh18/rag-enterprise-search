from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from rerankers.cross_encoder import CrossEncoderReranker
from bandits.bandit_reranker import BanditReranker

class RetrievalPipeline:
    def __init__(self, use_cross_encoder=True, use_bandit=True, top_k=30):
        """Initialize the retrieval pipeline with optional rerankers"""
        self.embedder = Embedder()
        self.memory = MemoryStore(collection_name="documents")
        self.cross_encoder = CrossEncoderReranker() if use_cross_encoder else None # Cross-encoder for reranking
        self.bandit_reranker = BanditReranker() if use_bandit else None # Bandit-based reranker
        self.top_k = top_k # No of initial chunks to retrieve before reranking
    
    def retrieve(self, query, final_k=10):
        """Retrieve relevant chunks for a given query, applying rerankers if available"""
        query_embed = self.embedder.get_embedding(query)
        raw_results = self.memory.retrieve_chunks(query_embedding=query_embed, top_k=self.top_k)

        candidates = [
            {"text": chunk["text"], "meta": chunk["metadata"], "score": chunk["score"]}
            for chunk in raw_results
        ]
        if self.cross_encoder:
            candidates = self.cross_encoder.rerank(query, candidates)
        if self.bandit_reranker:
            candidates = self.bandit_reranker.rerank(query, candidates)
        return candidates[:final_k]