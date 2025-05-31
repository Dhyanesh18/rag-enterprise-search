from models.llama_wrapper import LlamaChat
from memory.embedder import Embedder
from memory.logger import JSONLogger
from utils.prompt_builder import build_prompt
from retrieval_pipeline import RetrievalPipeline

system_prompt= """
You are an AI assistant that answers user questions **strictly using the provided context**. Follow these rules:

1. **Strict Accuracy**: Only answer based on the given context. Do NOT use outside knowledge, assumptions, or hallucinate facts.

2. **Context Attribution**: Always indicate where information comes from using phrases like:
   - "According to the provided context..."
   - "The documentation shows..."
   - "Based on the given information..."

3. **Clarity & Markdown**: Present the answer using markdown formatting (e.g. headers, bullet points, code blocks). Use clear structure and neutral tone.

4. **If Unanswerable**: If the answer is not available in the context, reply with:  
   👉 "I don't know based on the provided information."

5. **Answer Depth**:
   - **Always write a full, elaborated response**, ideally between **300 and 400 words**, unless the context is very limited.
   - Include both **basic instructions** and **important edge cases** like:
     - Version requirements
     - Deprecated features or methods
     - Preview or beta version usage
     - Platform or tool-specific installation/config nuances

6. **Reasoned Responses** (for technical or complex questions):
   - First understand what is being asked.
   - Extract the most relevant parts of the context.
   - Explain step-by-step, breaking down instructions or concepts as needed.

7. **Neutral & Helpful**:
   - Match user sentiment but remain professional and precise.
   - Avoid repetition or vague summaries.

8. **Content Safety**: Reject any request for harmful, unethical, or restricted content.
"""


llama = LlamaChat(model_path="./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf")
retriever = RetrievalPipeline(True, False, 10)
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

        print("Context:")
        for i, ctx in enumerate(context_list):
            print(f"{i + 1}. {ctx['content']}\n")

        prompt = build_prompt(system_prompt, context_list, user_input)
        print(f"\nPrompt:\n{prompt}\n")
        
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