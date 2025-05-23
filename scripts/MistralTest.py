from llama_cpp import Llama

# Initialize the model
llm = Llama(
    model_path="./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    n_gpu_layers=24,
    n_threads=12,
    n_ctx=4096
)

system_prompt = (
    "You are a helpful AI assistant. Answer concisely and clearly."
)
chat_history = f"<|system|>\n{system_prompt}\n"

print("Chat with Mistral (type Ctrl+C to stop, or type /reset to clear chat)")

try:
    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "/reset":
            chat_history = f"<|system|>\n{system_prompt}\n"
            print("Chat history reset.")
            continue

        # Format prompt with role tags
        chat_history += f"<|user|>\n{user_input}\n<|assistant|>\n"

        # Generate response
        response = llm(chat_history, max_tokens=200, stop=["<|user|>", "<|assistant|>", "<|system|>"])
        answer = response["choices"][0]["text"].strip()

        print(f"Assistant: {answer}\n")
        chat_history += f"{answer}\n"

except KeyboardInterrupt:
    print("\nExiting chat. Bye!")
