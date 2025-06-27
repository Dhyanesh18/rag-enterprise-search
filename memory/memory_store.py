import time
import chromadb
from chromadb.config import Settings
import hashlib

class MemoryStore:
    """
    MemoryStore provides a persistent vector store using ChromaDB
    to store and retrieve document chunks based on semantic embeddings.
    """
    def __init__(self, persist_dir="./chroma_store", collection_name="documents"):
        """
        Initializes the MemoryStore with a persistent ChromaDB client.

        Args:
            persist_dir (str): Directory path where Chroma will persist data.
            collection_name (str): Name of the Chroma collection.
        """
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def retrieve_chunks(self, query_embedding, top_k=8, filters=None, include_scores=True):
         """
        Retrieves top-k most similar chunks from the collection based on a query embedding.

        Args:
            query_embedding (list): The vector representation of the query.
            top_k (int): Number of top results to return. Defaults to 8.
            filters (dict): Optional metadata filters (e.g., {"file_type": "pdf"}).
            include_scores (bool): Whether to include similarity scores in results.

        Returns:
            list: A list of dictionaries with keys 'text', 'metadata', and 'score'.
        """
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
        """
        Deletes the existing collection and recreates a new one with the same name.
        Useful for clearing all stored data.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            print(f"[Info] Collection '{self.collection_name}' did not exist: {e}")

        self.collection = self.client.get_or_create_collection(name=self.collection_name)



    def store_document(self, chunk, embedding, file_path, file_type, metadata=None):
        """
        Stores a single document chunk along with its embedding and metadata.

        Args:
            chunk (str): The document text chunk to store.
            embedding (list): The vector embedding for the chunk.
            file_path (str): Original file path of the document.
            file_type (str): File type (e.g., ".pdf", ".txt").
            metadata (dict): Additional metadata (optional).
        """
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
