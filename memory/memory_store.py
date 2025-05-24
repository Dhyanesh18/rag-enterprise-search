import time
import chromadb
from chromadb.config import Settings

class MemoryStore:
    def __init__(self, persist_dir="./chroma_store", collection_name="chat_history"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def store(self, user_msg, assistant_msg, embedding):
        combined = f"{user_msg} ||| {assistant_msg}"
        self.collection.add(documents=[combined], embeddings=[embedding], ids=[str(time.time())])

    def retrieve(self, query_embedding, top_k=3):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        context = []
        for doc in results["documents"][0]:
            parts = doc.split("|||")
            if len(parts) != 2:
                continue 
            user, assistant = parts
            context.append({
                "user": user.strip(),
                "assistant": assistant.strip()
            })
        return context

    def reset(self):
        self.client.delete_collection("chat_history")
        self.collection = self.client.get_or_create_collection(name="chat_history")
