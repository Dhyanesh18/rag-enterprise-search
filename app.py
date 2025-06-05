from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from models.llama_wrapper import LlamaChat
from memory.logger import JSONLogger
from hybrid_retrieval_pipeline import HybridRetrievalPipeline
from utils.pipeline_runner import run_hyde_pipeline, run_standard_pipeline
from utils.prompt_builder import build_prompt

# Global variables for models and pipeline
llama = None
pipeline = None
logger = None

DATA_DIR = "./data"
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.docx', '.pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024 #10MB

system_prompt = """
You are a context-only assistant. NEVER use your training knowledge.
Rules:
- ONLY use the provided context to answer
- If the context doesn't contain the answer, respond: "I don't know based on the provided context"
- For technical questions, include ALL details from the context including version requirements, warnings, and edge cases
- If the context is not relevant, respond: "I don't know based on the provided context"
- If the context is not sufficient, respond: "I don't know based on the provided context"
- Context 1 is always the most relevant, so prioritize it, followed by Context 2, and so on
- Use the format: "Based on the context: [your answer]"
Context will be provided after this prompt.
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
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
    
    yield  # This is where the application runs
    
    # Shutdown
    print("Shutting down...")
    if llama:
        del llama
        print("Models cleaned up")

# Initialize FastAPI app with lifespan
app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

# Mount static files for serving HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

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

async def hybrid_retrieve_with_hyde(query, llm, pipeline):
    """Run the hybrid retrieval pipeline with HyDE and standard retrieval in parallel."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        task1 = loop.run_in_executor(executor, run_standard_pipeline, query, pipeline)
        task2 = loop.run_in_executor(executor, run_hyde_pipeline, query, llm, pipeline)
        results_std, results_hyde = await asyncio.gather(task1, task2)
    
    # Final fusion of both results using Reciprocal Rank Fusion (RRF)
    if not results_std and not results_hyde:
        return []
    
    fused = pipeline.reciprocal_rank_fusion([results_std, results_hyde])
    return fused[:10]

def run_bulk_ingest():
    """Run the Bulk ingestion script"""
    try:
        result = subprocess.run(
            ["python", "bulk_ingest.py"],
            capture_output = True,
            text = True,
            cwd = os.getcwd()
        )
        if result.returncode == 0:
            return True, result.stdout
        else: 
            return False, result.stderr
    except Exception as e:
        return False, str(e)
        

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

@app.post("/upload", response_class=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the data directory and trigger ingestion"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024*10)}MB"
            )
        
        # Save the file
        file_path = os.path.join(DATA_DIR, file.filename)
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_path)
            file_path = f"{name}_{counter}{ext}"
            counter+=1

        with open(file_path, "wb") as f:
            f.write(content)

        success, output = run_bulk_ingest()

        if success:
            return UploadResponse(
                success = True,
                message = f"Document uploaded and processes successfully. {output.strip()}",
                filename = os.path.basename(file_path)
            )
        else:
            return UploadResponse(
                success = False,
                message = f"Document upload and processing failed",
                filename = os.path.basename(file_path),
                error = output
            )
        
    except HTTPException:
        raise
    except Exception as e:
            return UploadResponse(
                success = False,
                message = "Upload Failed",
                error = str(e)
            )




@app.get("/documents")
async def list_documents():
    """List all documents in the data directory"""
    try:
        files = []
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
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
    """Delete a document from the data directory"""
    try:
        file_path = os.path.join(DATA_DIR, filename)

        abs_data_dir = os.path.abspath(DATA_DIR)
        abs_file_path = os.path.abspath(file_path)

        if not abs_file_path.startswith(abs_data_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")

        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)

        success, output = run_bulk_ingest()

        return {
            "success": True,
            "message": f"Document {filename} deleted successfully",
            "reindex_success": success,
            "reindex_output": output if success else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/reindex")
async def reindex_documents():
    """Manually reindex all the documents"""
    try:
        success, output = run_bulk_ingest()

        return {
            "success" : success,
            "message" : "Re-indexing completed" if success else "Re-indexing failed",
            "output": output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Main chat endpoint"""
    try:
        if not message.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Run the hybrid retrieval pipeline
        if message.message in ["Explain the main concepts from the documents", "What are the most important topics covered?", "Can you help me understand specific details about...", "Show me related information about..."]:
            top_chunks = run_standard_pipeline(message.message, pipeline)
        else:    
            top_chunks = await hybrid_retrieve_with_hyde(message.message, llama, pipeline)
        
        context_list = [
            {
                "content": chunk["text"],
                "score": float(chunk.get("score")) if chunk.get("score") is not None else None,
                "meta": chunk.get("meta")
            }
            for chunk in top_chunks
        ]
        
        if not context_list:
            return ChatResponse(
                response="No relevant context found. Please try a different question.",
                context=[],
                success=False,
                error="No relevant context found"
            )
        
        # Build prompt and generate response
        prompt = build_prompt(system_prompt, context_list, message.message)
        
        try:
            response = llama.generate(prompt)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
            print(f"Generation error: {e}")
            return ChatResponse(
                response=response,
                context=context_list,
                success=False,
                error=str(e)
            )
        
        # Log the interaction
        logger.log(message.message, response)
        
        return ChatResponse(
            response=response,
            context=context_list,
            success=True
        )
        
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": llama is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)