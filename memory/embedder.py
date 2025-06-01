from sentence_transformers import SentenceTransformer

class Embedder:
    """ 
    Embedder class for generating text embeddings using SentenceTransformer.
    Uses the 'all-MiniLM-L6-v2' model by default, which is efficient and effective for many tasks.
    For specific use cases models like multi-qa-minilm-l6-v2 or all-mpnet-base-v2 can be used.
    all-mpnet-base-v2 is more powerful but requires more resources.
    Considering the trade-off between performance and latency, resource constraints, all-MiniLM-L6-v2 is a good default choice.
    You can change the model by passing a different model name to the constructor.
    """
    def __init__(self, model_path="./models/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_path)

    def get_embedding(self, text: str):
        return self.model.encode([text])[0].tolist()
