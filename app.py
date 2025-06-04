from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

supported_extensions = {'.txt', '.md', '.docx', '.pdf'}
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

@app.get("/upload", response_class=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    pass


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Main chat endpoint"""
    try:
        if not message.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Run the hybrid retrieval pipeline
        top_chunks = await hybrid_retrieve_with_hyde(message.message, llama, pipeline)
        
        context_list = [
            {
                "content": chunk["text"],
                "score": chunk.get("score"),
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