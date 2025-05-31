from models.llama_wrapper import LlamaChat
from memory.embedder import Embedder
from memory.logger import JSONLogger
from utils.prompt_builder import build_prompt
from retrieval_pipeline import RetrievalPipeline
from hybrid_retrieval_pipeline import HybridRetrievalPipeline

system_prompt= """
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


llama = LlamaChat(model_path="./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf")
# retriever = RetrievalPipeline(use_cross_encoder=True, use_bandit=False, top_k=10)
retriever = HybridRetrievalPipeline(use_cross_encoder=True, use_bandit=False, top_k=50)
embedder = Embedder()
logger = JSONLogger()

print("Chat with Mistral (type Ctrl+C to stop)")

try:
    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        top_chunks = retriever.retrieve(user_input)

        context_list = [  
            {
                "user": "REFERENCE",
                "content": chunk["text"],
                "score": chunk.get("score"),
                "meta": chunk.get("meta")
            }   
            for chunk in top_chunks
        ]   

        if not context_list:
            print("No relevant context found. Please try again.")
            continue

        for i, ctx in enumerate(context_list):
            print(f"{i + 1}. {ctx['content']}\n")

        prompt = build_prompt(system_prompt, context_list, user_input)
        
        try:
            response = llama.generate(prompt)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
            print(f"Generation error: {e}")

        print(f"Assistant: {response}\n")

        logger.log(user_input, response)

except KeyboardInterrupt:
    print("\nExiting chat. Bye!")

finally:
    del llama