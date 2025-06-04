from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from models.llama_wrapper import LlamaChat
from memory.logger import JSONLogger
from hybrid_retrieval_pipeline import HybridRetrievalPipeline
from utils.pipeline_runner import run_hyde_pipeline, run_standard_pipeline
from utils.prompt_builder import build_prompt

app = FastAPI(title="RAG Chatbot API")

# Mount static files for serving HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models and pipeline
llama = None
pipeline = None
logger = None

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

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context: list
    success: bool
    error: str = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and pipeline on startup"""
    global llama, pipeline, logger
    try:
        llama = LlamaChat(model_path="./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf")
        pipeline = HybridRetrievalPipeline(use_cross_encoder=True, top_k=50)
        logger = JSONLogger()
        print("Models and pipeline initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global llama
    if llama:
        del llama
        print("Models cleaned up")

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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

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