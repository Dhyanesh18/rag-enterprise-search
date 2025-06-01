import asyncio
from concurrent.futures import ThreadPoolExecutor
from models.llama_wrapper import LlamaChat
from memory.logger import JSONLogger
from hybrid_retrieval_pipeline import HybridRetrievalPipeline
from utils.pipeline_runner import run_hyde_pipeline, run_standard_pipeline
from utils.prompt_builder import build_prompt


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
pipeline = HybridRetrievalPipeline(use_cross_encoder=True, top_k=50)
logger = JSONLogger()


async def hybrid_retrieve_with_hyde(query, llm, pipeline):
    """Run the hybrid retrieval pipeline with HyDE and standard retrieval in parallel."""
    # Run both pipelines in parallel using asyncio and ThreadPoolExecutor
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


if __name__ == "__main__":
    print("Chat with Mistral (type Ctrl+C to stop)")

    loop = asyncio.get_event_loop()
    try:
        while True:
            user_input = input("User: ").strip()
            if not user_input:
                continue

            # Run the hybrid retrieval pipeline asynchronously
            top_chunks = loop.run_until_complete(hybrid_retrieve_with_hyde(user_input, llama, pipeline))

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

            print(f"Assistant: {response}")
            logger.log(user_input, response)

    except KeyboardInterrupt:
        print("\nExiting chat. Goodbye!")

    finally:
        del llama