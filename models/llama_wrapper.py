from llama_cpp import Llama
import os

class LlamaChat:
    def __init__(self, model_path, n_gpu_layers=20, n_threads=6, n_ctx=8192):
        """Initialize the Llama model with the optimized parameters for 4GB VRAM"""
        os.environ["GGML_CUDA_MMQ"] = "1"  # Enable memory-efficient kernels
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers, # No of layers to offload to GPU
            n_threads=n_threads, # Number of CPU threads to use
            n_ctx=n_ctx, # Context Size of the model
            offload_kqv=True  # Saves VRAM
        )
    
    def generate(self, prompt, max_tokens=1536, temperature=0.2):
        """Generate a response from the LLM based on the provided prompt"""
        self.llm.reset()  # Clear cache
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9, 
            top_k=40,
            repeat_penalty=1.1,
            stop=["<|user|>", "<|assistant|>", "<|/assistant|>", "<|system|>"],
            echo=False 
        )
        return output["choices"][0]["text"].strip()