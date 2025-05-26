import os
from PyPDF2 import PdfReader
import pdfplumber
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from memory.logger import JSONLogger
import re

embedder = Embedder()
logger = JSONLogger("document_ingestion_log.json")


def extract_pdf_text(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception:
        print(f"Failed to extract text from {file_path} using pdfplumber, trying PyPDF2...")
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    
def extract_md_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_txt_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def extract_docx_text(file_path):
    doc = Document(file_path)
    full_text = []

    # Extract text from para with special handling for headings
    for para in doc.paragraphs:
        if para.style.name.startswith('Heading') and para.text.strip():
            full_text.append(f"\n## {para.text.strip()}\n")
        elif para.text.strip():
            full_text.append(para.text.strip())

    # Extract text from each table
    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_text.append(cell_text)
            if row_text:
                full_text.append("\t".join(row_text))
    return "\n".join(full_text).strip()

def clean_text(text):
    # Normalize whitespace: multiple spaces/newlines → single space
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# Function to ingest a file and store its content in the memory store
def ingest_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = extract_pdf_text(file_path)
    elif ext == ".md":
        text = extract_md_text(file_path)
    elif ext == ".txt":
        text = extract_txt_text(file_path)
    elif ext == ".docx":
        text = extract_docx_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    total_chunks = len(chunks)
    ingested_chunks = 0

    store = MemoryStore(collection_name="documents")
    for chunk in chunks:
        if chunk.strip():
            embedding = embedder.get_embedding(chunk)
            store.store_document(chunk, embedding, file_path, ext)
            ingested_chunks += 1

    logger.log_ingestion(file_path, total_chunks, ingested_chunks)
    print(f"Ingested {ingested_chunks}/{total_chunks} chunks from {file_path}")