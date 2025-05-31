import os
import logging
import json
import csv
from datetime import datetime
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from hybrid_retrieval_pipeline import HybridRetrievalPipeline

logging.basicConfig(level=logging.INFO)

class BEIRHybridRetriever:
    """Wrapper to make HybridRetrievalPipeline compatible with BEIR evaluation format"""
    def __init__(self, use_cross_encoder=True, use_bandit=False, top_k=100):
        self.embedder = Embedder()
        self.memory = MemoryStore(persist_dir="./chroma_store_beir_hybrid")
        self.pipeline = HybridRetrievalPipeline(
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
        
        # After indexing all documents, rebuild the BM25 index for hybrid retrieval
        print("Rebuilding BM25 index for hybrid retrieval...")
        self.pipeline.rebuild_bm25_index()

    def retrieve(self, queries, top_k=15):
        """Retrieve using the hybrid RAG pipeline and convert to BEIR format"""
        print(f"Retrieving with hybrid pipeline (cross_encoder={self.pipeline.cross_encoder is not None}, final_k={top_k})")
        results = {}
        
        for query_id, query_text in tqdm(queries.items()):
            # Use the hybrid pipeline - this will return exactly final_k results
            retrieved = self.pipeline.retrieve(query_text, final_k=top_k)
            
            # Convert to BEIR expected format: {doc_id: score}
            scored = {}
            for doc in retrieved:
                doc_id = doc["meta"].get("doc_id")
                if doc_id:
                    # Use rerank_score if available (from cross-encoder), otherwise RRF score or original score
                    score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0)))
                    # BEIR expects higher scores = better, our pipeline already handles this
                    scored[doc_id] = float(score)
            
            results[query_id] = scored
        
        return results


def save_results_to_files(dataset_name, results_summary, timestamp):
    """Save evaluation results to JSON and CSV files"""
    # Create results directory if it doesn't exist
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare data for saving
    results_data = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "results": results_summary
    }
    
    # Save detailed JSON results
    json_filename = f"{results_dir}/{dataset_name}_hybrid_evaluation_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"Detailed results saved to: {json_filename}")
    
    # Save summary CSV
    csv_filename = f"{results_dir}/{dataset_name}_hybrid_summary_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["Configuration", "nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10", "nDCG@15",
                 "Recall@1", "Recall@3", "Recall@5", "Recall@10", "Recall@15",
                 "MAP@1", "MAP@3", "MAP@5", "MAP@10", "MAP@15",
                 "Precision@1", "Precision@3", "Precision@5", "Precision@10", "Precision@15"]
        writer.writerow(header)
        
        # Write data rows
        for config_name, metrics in results_summary.items():
            row = [config_name]
            
            # Add nDCG values
            for k in [1, 3, 5, 10, 15]:
                row.append(metrics["ndcg"].get(k, metrics["ndcg"].get(str(k), 0)))
            
            # Add Recall values
            for k in [1, 3, 5, 10, 15]:
                row.append(metrics["recall"].get(k, metrics["recall"].get(str(k), 0)))
            
            # Add MAP values
            for k in [1, 3, 5, 10, 15]:
                row.append(metrics["map"].get(k, metrics["map"].get(str(k), 0)))
            
            # Add Precision values
            for k in [1, 3, 5, 10, 15]:
                row.append(metrics["precision"].get(k, metrics["precision"].get(str(k), 0)))
            
            writer.writerow(row)
    
    print(f"Summary CSV saved to: {csv_filename}")
    
    # Save a simple comparison CSV for quick analysis
    comparison_filename = f"{results_dir}/{dataset_name}_hybrid_comparison_{timestamp}.csv"
    with open(comparison_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "nDCG@10", "Recall@10", "MAP@10"])
        
        for config_name, metrics in results_summary.items():
            ndcg_10 = metrics["ndcg"].get(10, metrics["ndcg"].get("10", 0))
            recall_10 = metrics["recall"].get(10, metrics["recall"].get("10", 0))
            map_10 = metrics["map"].get(10, metrics["map"].get("10", 0))
            writer.writerow([config_name, f"{ndcg_10:.4f}", f"{recall_10:.4f}", f"{map_10:.4f}"])
    
    print(f"Quick comparison CSV saved to: {comparison_filename}")


def evaluate_beir_dataset(dataset_name, dataset_path, configurations):
    """Evaluate dataset with different hybrid pipeline configurations"""
    print(f"\n=== Loading {dataset_name} dataset ===")
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")
    print(f"Loaded {len(corpus)} docs, {len(queries)} queries")
    
    results_summary = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for config_name, config in configurations.items():
        print(f"\n=== Evaluating {config_name} ===")
        
        retriever = BEIRHybridRetriever(
            use_cross_encoder=config["cross_encoder"],
            use_bandit=config["bandit"],
            top_k=config["top_k"]
        )
        
        # Index once per configuration (in case we want different stores)
        retriever.index(corpus)
        
        # Retrieve and evaluate - using final_k=15 for all configurations
        results = retriever.retrieve(queries, top_k=15)
        
        evaluator = EvaluateRetrieval()
        # Evaluate at ranks that make sense for our final_k=15
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 15])
        
        # Store results
        results_summary[config_name] = {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
            "config": config  # Store configuration for reference
        }
        
        print(f"\nResults for {config_name}:")
        for k in [1, 3, 5, 10, 15]:
            try:
                print(f"  nDCG@{k}: {ndcg[k]:.4f} | Recall@{k}: {recall[k]:.4f} | Precision@{k}: {precision[k]:.4f}")
            except KeyError:
                print(f"  (No data at rank @{k})")

    
    # Print comparison
    print(f"\n=== COMPARISON SUMMARY for {dataset_name} ===")
    print("Configuration".ljust(30) + "nDCG@10".ljust(12) + "Recall@10".ljust(12) + "MAP@10")
    print("-" * 65)
    
    for config_name, metrics in results_summary.items():
        ndcg_10 = metrics["ndcg"].get(10) or metrics["ndcg"].get("10", 0)
        recall_10 = metrics["recall"].get(10) or metrics["recall"].get("10", 0)
        map_10 = metrics["map"].get(10) or metrics["map"].get("10", 0)

        print(f"{config_name:<30}{ndcg_10:<12.4f}{recall_10:<12.4f}{map_10:.4f}")
    
    # Save results to files
    save_results_to_files(dataset_name, results_summary, timestamp)
    
    return results_summary


if __name__ == "__main__":
    DATASETS = {
        # "nq": "./datasets/nq"
        # "msmarco": "./datasets/msmarco",
        "scifact": "./datasets/scifact"
    }
    
    # Different configurations to test hybrid retrieval (no bandit learning)
    CONFIGURATIONS = {
    # Baseline - just hybrid retrieval, no cross-encoder
        "hybrid_baseline": {
            "cross_encoder": False,
            "bandit": False,
            "top_k": 50
        },
    
        # With cross-encoder - to measure cross-encoder impact
        "hybrid_with_cross_encoder": {
            "cross_encoder": True,
            "bandit": False,
            "top_k": 50  # Same top_k for fair comparison
        }
    }

    all_results = {}
    
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"Dataset path missing: {path}")
            continue
        
        dataset_results = evaluate_beir_dataset(name, path, CONFIGURATIONS)
        all_results[name] = dataset_results
    
    # Save combined results across all datasets
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        combined_filename = f"{results_dir}/combined_hybrid_evaluation_{timestamp}.json"
        combined_data = {
            "timestamp": timestamp,
            "datasets": all_results,
            "configurations": CONFIGURATIONS
        }
        
        with open(combined_filename, 'w') as f:
            json.dump(combined_data, f, indent=2, default=str)
        
        print(f"\n=== EVALUATION COMPLETE ===")
        print(f"Combined results saved to: {combined_filename}")
        print(f"Individual dataset results saved in: {results_dir}/")
        print(f"Check the CSV files for easy analysis and comparison!")