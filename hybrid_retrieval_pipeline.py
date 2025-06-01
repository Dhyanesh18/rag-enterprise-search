from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from rerankers.cross_encoder import CrossEncoderReranker
from rank_bm25 import BM25Okapi
import pickle
import os
from typing import List, Dict, Any

class HybridRetrievalPipeline:
    def __init__(self, use_cross_encoder=True, top_k=50):
        """Initialize the hybrid retrieval pipeline with BM25 + Dense -> RRF"""
        self.embedder = Embedder()
        self.memory = MemoryStore(collection_name="documents")
        self.cross_encoder = CrossEncoderReranker() if use_cross_encoder else None
        self.top_k = top_k
        
        # BM25 components
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []
        self._load_or_build_bm25_index()
    
    def _load_or_build_bm25_index(self):
        """Load existing BM25 index or build it from ChromaDB"""
        bm25_path = "bm25_index.pkl"
        docs_path = "bm25_docs.pkl"
        meta_path = "bm25_meta.pkl"
        
        if all(os.path.exists(p) for p in [bm25_path, docs_path, meta_path]):
            print("Loading existing BM25 index...")
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            with open(docs_path, 'rb') as f:
                self.bm25_documents = pickle.load(f)
            with open(meta_path, 'rb') as f:
                self.bm25_metadata = pickle.load(f)
            print(f"Loaded BM25 index with {len(self.bm25_documents)} documents")
        else:
            print("Building BM25 index from ChromaDB...")
            self._build_bm25_index()
            
    def _build_bm25_index(self):
        """Build BM25 index from all documents in ChromaDB"""
        # Get all documents from ChromaDB
        try:
            results = self.memory.collection.get()
            documents = results['documents']
            metadatas = results['metadatas']
            
            if not documents:
                print("No documents found in ChromaDB for BM25 indexing")
                return
                
            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in documents]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_documents = documents
            self.bm25_metadata = metadatas
            
            # Save the index
            with open("bm25_index.pkl", 'wb') as f:
                pickle.dump(self.bm25_index, f)
            with open("bm25_docs.pkl", 'wb') as f:
                pickle.dump(self.bm25_documents, f)
            with open("bm25_meta.pkl", 'wb') as f:
                pickle.dump(self.bm25_metadata, f)
                
            print(f"Built and saved BM25 index with {len(documents)} documents")
            
        except Exception as e:
            print(f"Error building BM25 index: {e}")
            self.bm25_index = None
    
    def _bm25_search(self, query: str, top_k: int = 25) -> List[Dict[str, Any]]:
        """Search using BM25"""
        if not self.bm25_index:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top results
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append({
                    "text": self.bm25_documents[idx],
                    "meta": self.bm25_metadata[idx],
                    "score": float(scores[idx]),
                    "retrieval_method": "bm25"
                })
        
        return results
    
    def _dense_search(self, query: str, top_k: int = 25) -> List[Dict[str, Any]]:
        """Search using dense embeddings"""
        query_embed = self.embedder.get_embedding(query)
        raw_results = self.memory.retrieve_chunks(query_embedding=query_embed, top_k=top_k)
        
        results = []
        for chunk in raw_results:
            # Convert distance to similarity score (ChromaDB returns distances)
            similarity_score = 1.0 / (1.0 + chunk["score"])
            results.append({
                "text": chunk["text"],
                "meta": chunk["metadata"],
                "score": similarity_score,
                "retrieval_method": "dense"
            })
        
        return results
    
    def _reciprocal_rank_fusion(self, result_lists: List[List[Dict]], k: int = 60) -> List[Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion to combine multiple result lists"""
        fused_scores = {}
        
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                doc_id = hash(doc["text"])  # Simple doc ID based on content
                
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {
                        "doc": doc,
                        "score": 0.0,
                        "methods": []
                    }
                
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank + 1)
                fused_scores[doc_id]["score"] += rrf_score
                fused_scores[doc_id]["methods"].append(doc["retrieval_method"])
        
        # Sort by fused score and return documents
        sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        
        final_results = []
        for item in sorted_results:
            doc = item["doc"].copy()
            doc["rrf_score"] = item["score"]
            doc["retrieval_methods"] = list(set(item["methods"]))  # Remove duplicates
            doc["score"] = item["score"]  # Use RRF score as primary score
            final_results.append(doc)
        
        return final_results
    
    def retrieve(self, query: str, final_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using hybrid approach: BM25 + Dense + RRF + Reranking"""
        
        # Step 1: Retrieve from both systems
        dense_results = self._dense_search(query, top_k=self.top_k//2)
        bm25_results = self._bm25_search(query, top_k=self.top_k//2)
        
        print(f"Dense retrieval: {len(dense_results)} results")
        print(f"BM25 retrieval: {len(bm25_results)} results")
        
        # Step 2: Apply RRF fusion
        if bm25_results and dense_results:
            fused_results = self._reciprocal_rank_fusion([dense_results, bm25_results])
        elif dense_results:
            fused_results = dense_results
        elif bm25_results:
            fused_results = bm25_results
        else:
            return []
        
        print(f"After RRF fusion: {len(fused_results)} results")
        
        # Step 3: Take top candidates for potential reranking
        candidates = fused_results[:self.top_k]
        
        # Step 4: Apply rerankers if available
        if self.cross_encoder:
            candidates = self.cross_encoder.rerank(query, candidates, top_k=len(candidates))
            print(f"After cross-encoder reranking: {len(candidates)} results")
        
        return candidates[:final_k]
    
    def rebuild_bm25_index(self):
        """Force rebuild of BM25 index (call this after adding new documents)"""
        print("Rebuilding BM25 index...")
        # Remove existing files
        for file in ["bm25_index.pkl", "bm25_docs.pkl", "bm25_meta.pkl"]:
            if os.path.exists(file):
                os.remove(file)
        
        # Rebuild
        self._build_bm25_index()