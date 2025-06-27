"""
RAG Chatbot API (FastAPI-based)

This application allows users to upload documents, which are processed into embeddings
and stored for retrieval. Users can then query the system, which performs hybrid retrieval
(using standard + HyDE) and returns LLM-generated responses based only on the document context.

Features:
- Document upload with type/size validation
- Auto ingestion and re-indexing
- Hybrid retrieval using reciprocal rank fusion
- Context-only LLM responses
- Static frontend serving
"""

from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Custom modules
from models.llama_wrapper import LlamaChat
from memory.logger import JSONLogger
from hybrid_retrieval_pipeline import HybridRetrievalPipeline
from utils.pipeline_runner import run_hyde_pipeline, run_standard_pipeline
from utils.prompt_builder import build_prompt

# === Global Variables ===
llama = None
pipeline = None
logger = None

DATA_DIR = "./data"
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.docx', '.pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# === System Prompt (included in every LLM call) ===
system_prompt = """
You are a context-only assistant. NEVER use your training knowledge.
Rules:
- ONLY use the provided context to answer
- If the context doesn't contain the answer, respond: "I don't know based on the provided context"
- For technical questions, include ALL details from the context including version requirements, warnings, and edge cases
- Context 1 is always the most relevant, so prioritize it, followed by Context 2, and so on
- Use the format: "Based on the context: [your answer]"
"""

# === Lifespan Event Handler (startup/shutdown) ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown of global resources like LLM and retrieval pipeline.
    """
    global llama, pipeline, logger
    try:
        print("Initializing models and pipeline...")
        llama = LlamaChat(model_path="./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf")
        pipeline = HybridRetrievalPipeline(use_cross_encoder=True, top_k=50)
        logger = JSONLogger()
        print("Models and pipeline initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise e

    yield  # App runs here

    print("Shutting down...")
    if llama:
        del llama
        print("Models cleaned up")


# === App Setup ===
app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Request/Response Models ===
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context: list
    success: bool
    error: str = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str = None
    error: str = None


# === Helper Functions ===

async def hybrid_retrieve_with_hyde(query, llm, pipeline):
    """
    Performs both standard and HyDE retrieval in parallel, and fuses results using RRF.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        task1 = loop.run_in_executor(executor, run_standard_pipeline, query, pipeline)
        task2 = loop.run_in_executor(executor, run_hyde_pipeline, query, llm, pipeline)
        results_std, results_hyde = await asyncio.gather(task1, task2)

    if not results_std and not results_hyde:
        return []

    fused = pipeline.reciprocal_rank_fusion([results_std, results_hyde])
    return fused[:10]  # return top 10 results


def run_bulk_ingest():
    """
    Triggers document ingestion via `bulk_ingest.py` using subprocess.
    """
    try:
        result = subprocess.run(
            ["python", "bulk_ingest.py"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        return (result.returncode == 0), (result.stdout if result.returncode == 0 else result.stderr)
    except Exception as e:
        return False, str(e)


# === API Routes ===

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main frontend HTML page."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    return FileResponse("static/favicon.ico")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for ingestion. Supported: .pdf, .docx, .md, .txt
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")

        # Avoid name collision by appending counter
        file_path = os.path.join(DATA_DIR, file.filename)
        counter = 1
        while os.path.exists(file_path):
            name, ext = os.path.splitext(file.filename)
            file_path = os.path.join(DATA_DIR, f"{name}_{counter}{ext}")
            counter += 1

        with open(file_path, "wb") as f:
            f.write(content)

        success, output = run_bulk_ingest()
        return UploadResponse(
            success=success,
            message="Document uploaded and processed." if success else "Upload succeeded but ingestion failed.",
            filename=os.path.basename(file_path),
            error=None if success else output
        )
    except Exception as e:
        return UploadResponse(success=False, message="Upload Failed", error=str(e))


@app.get("/documents")
async def list_documents():
    """Returns a list of documents stored in the data directory."""
    try:
        files = []
        for filename in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, filename)
            if os.path.isfile(path):
                stat = os.stat(path)
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Deletes a document and triggers reindexing."""
    try:
        file_path = os.path.join(DATA_DIR, filename)

        # Safety check: prevent path traversal
        if not os.path.abspath(file_path).startswith(os.path.abspath(DATA_DIR)):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        os.remove(file_path)
        success, output = run_bulk_ingest()

        return {
            "success": True,
            "message": f"{filename} deleted.",
            "reindex_success": success,
            "reindex_output": output if success else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
async def reindex_documents():
    """Manually re-runs the ingestion process on all files in `./data`."""
    try:
        success, output = run_bulk_ingest()
        return {
            "success": success,
            "message": "Reindexing completed." if success else "Reindexing failed.",
            "output": output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """
    Main chat endpoint. Performs hybrid retrieval and uses local LLM to generate a context-grounded answer.
    """
    try:
        if not message.message.strip():
            raise HTTPException(status_code=400, detail="Empty message.")

        # For general queries, fallback to standard pipeline
        if message.message in [
            "Explain the main concepts from the documents",
            "What are the most important topics covered?",
            "Can you help me understand specific details about...",
            "Show me related information about..."
        ]:
            top_chunks = run_standard_pipeline(message.message, pipeline)
        else:
            top_chunks = await hybrid_retrieve_with_hyde(message.message, llama, pipeline)

        if not top_chunks:
            return ChatResponse(
                response="No relevant context found. Please try a different question.",
                context=[],
                success=False,
                error="No context"
            )

        # Format prompt for generation
        context_list = [
            {
                "content": chunk["text"],
                "score": float(chunk.get("score")) if chunk.get("score") else None,
                "meta": chunk.get("meta")
            }
            for chunk in top_chunks
        ]
        prompt = build_prompt(system_prompt, context_list, message.message)

        # Generate response using local LLM
        try:
            response = llama.generate(prompt)
        except Exception as e:
            return ChatResponse(
                response=f"Error generating response: {str(e)}",
                context=context_list,
                success=False,
                error=str(e)
            )

        logger.log(message.message, response)
        return ChatResponse(response=response, context=context_list, success=True)

    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Returns server and model readiness info."""
    return {"status": "healthy", "models_loaded": llama is not None}


# === Run the Server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
