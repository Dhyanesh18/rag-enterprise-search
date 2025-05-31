import os
import logging
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from retrieval_pipeline import RetrievalPipeline

logging.basicConfig(level=logging.INFO)

class BEIRRetriever:
    """Wrapper to make RetrievalPipeline compatible with BEIR evaluation format"""
    def __init__(self, use_cross_encoder=True, use_bandit=False, top_k=100):
        self.embedder = Embedder()
        self.memory = MemoryStore(persist_dir="./chroma_store_beir")
        self.pipeline = RetrievalPipeline(
            use_cross_encoder=use_cross_encoder, 
            use_bandit=use_bandit, 
            top_k=top_k
        )
        # Override the pipeline's memory store to use our BEIR-specific one
        self.pipeline.memory = self.memory

    def index(self, corpus):
        """Index corpus into vector store with BEIR format"""
        print("Indexing corpus into vector store...")
        self.memory.reset()
        
        for doc_id, content in tqdm(corpus.items()):
            combined_text = (content.get("title", "") + " " + content.get("text", "")).strip()
            embedding = self.embedder.get_embedding(combined_text)
            metadata = {"doc_id": doc_id}
            self.memory.store_document(
                chunk=combined_text,
                embedding=embedding,
                file_path=doc_id,
                file_type="beir",
                metadata=metadata
            )

    def retrieve(self, queries, top_k=100):
        """Retrieve using the full RAG pipeline and convert to BEIR format"""
        print(f"Retrieving with pipeline (cross_encoder={self.pipeline.cross_encoder is not None}, top_k={top_k})")
        results = {}
        
        for query_id, query_text in tqdm(queries.items()):
            # Use the full pipeline
            retrieved = self.pipeline.retrieve(query_text, final_k=top_k)
            
            # Convert to BEIR expected format: {doc_id: score}
            scored = {}
            for doc in retrieved:
                doc_id = doc["meta"].get("doc_id")
                if doc_id:
                    # Use rerank_score if available (from cross-encoder), otherwise original score
                    score = doc.get("rerank_score", doc.get("score", 0))
                    # BEIR expects higher scores = better, our pipeline already handles this
                    scored[doc_id] = float(score)
            
            results[query_id] = scored
        
        return results


def evaluate_beir_dataset(dataset_name, dataset_path, configurations):
    """Evaluate dataset with different pipeline configurations"""
    print(f"\n=== Loading {dataset_name} dataset ===")
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")
    print(f"Loaded {len(corpus)} docs, {len(queries)} queries")
    
    results_summary = {}
    
    for config_name, config in configurations.items():
        print(f"\n=== Evaluating {config_name} ===")
        
        retriever = BEIRRetriever(
            use_cross_encoder=config["cross_encoder"],
            use_bandit=config["bandit"],
            top_k=config["top_k"]
        )
        
        # Index once per configuration (in case we want different stores)
        retriever.index(corpus)
        
        # Retrieve and evaluate
        results = retriever.retrieve(queries, top_k=100)
        
        evaluator = EvaluateRetrieval()
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])
        
        # Store results
        results_summary[config_name] = {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision
        }
        
        print(f"\nResults for {config_name}:")
        for k in [1, 3, 5, 10, 100]:
            try:
                print(f"  nDCG@{k}: {ndcg[k]:.4f} | Recall@{k}: {recall[k]:.4f} | Precision@{k}: {precision[k]:.4f}")
            except KeyError:
                print(f"  (No data at rank @{k})")

    
    # Print comparison
    print(f"\n=== COMPARISON SUMMARY for {dataset_name} ===")
    print("Configuration".ljust(25) + "nDCG@10".ljust(12) + "Recall@10".ljust(12) + "MAP@10")
    print("-" * 60)
    
    for config_name, metrics in results_summary.items():
        ndcg_10 = metrics["ndcg"].get(10) or metrics["ndcg"].get("10", 0)
        recall_10 = metrics["recall"].get(10) or metrics["recall"].get("10", 0)
        map_10 = metrics["map"].get(10) or metrics["map"].get("10", 0)

        print(f"{config_name:<25}{ndcg_10:<12.4f}{recall_10:<12.4f}{map_10:.4f}")


if __name__ == "__main__":
    DATASETS = {
        # "nq": "./datasets/nq"
        # "msmarco": "./datasets/msmarco",
        "scifact": "./datasets/scifact"
    }
    
    # Different configurations to test
    CONFIGURATIONS = {
        "baseline_embedding": {
            "cross_encoder": False,
            "bandit": False,
            "top_k": 100
        },
        "with_cross_encoder": {
            "cross_encoder": True,
            "bandit": False,
            "top_k": 100
        },
        
        # "with_bandit": {
        #     "cross_encoder": False,
        #     "bandit": True,
        #     "top_k": 100
        # },
        # "full_pipeline": {
        #     "cross_encoder": True,
        #     "bandit": True,
        #     "top_k": 100
        # }
    }

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"Dataset path missing: {path}")
            continue
        
        evaluate_beir_dataset(name, path, CONFIGURATIONS)