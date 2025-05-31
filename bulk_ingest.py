import os
from ingestion.document_ingestor import ingest_file

def ingest_all_documents(data_dir="./data"):
    """Ingest all documents from the specified directory into the memory store"""
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            try:
                print(f"Ingesting {fpath}...")
                ingest_file(fpath)
                print(f"Successfully ingested {fpath}")
            except Exception as e:
                print(f"Failed to ingest {fpath}: {e}")

if __name__ == "__main__":
    print("Starting bulk document ingestion...")
    ingest_all_documents()
    