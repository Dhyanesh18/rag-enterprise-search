import time
import chromadb
from chromadb.config import Settings
import hashlib

class MemoryStore:
    def __init__(self, persist_dir="./chroma_store", collection_name="documents"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def retrieve_chunks(self, query_embedding, top_k=8, filters=None, include_scores=True):
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }
        if filters:
            query_args["where"] = filters

        results = self.collection.query(**query_args)

        retrieved = []
        for doc, meta, score in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0] if include_scores else [None] * top_k
        ):
            retrieved.append({
                "text": doc,
                "metadata": meta,
                "score": score  # This is cosine distance (lower = better)
            })

        return retrieved


    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            print(f"[Info] Collection '{self.collection_name}' did not exist: {e}")

        self.collection = self.client.get_or_create_collection(name=self.collection_name)



    def store_document(self, chunk, embedding, file_path, file_type, metadata=None):
        chunk_id = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
    
        # Start with default metadata
        meta = {
            "source": file_path,
            "file_type": file_type.lstrip('.')
        }

        if metadata:
            meta.update(metadata)

        self.collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[chunk_id],
            metadatas=[meta]
        )
