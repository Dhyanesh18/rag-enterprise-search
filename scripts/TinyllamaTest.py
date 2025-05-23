from llama_cpp import Llama

llm = Llama(model_path="./models/tinyllama.gguf", n_gpu_layers=-1, verbose=False)

print("Chat with TinyLlama (type Ctrl+C to stop)")

try:
    while True:
        user_input = input("User: ")
        if not user_input.strip():
            continue
        prompt = f"Q: {user_input}\nA:"
        response = llm(prompt, max_tokens=100, stop=["\nQ:", "\nA:"])
        answer = response['choices'][0]['text'].strip()
        print(f"TinyLlama: {answer}\n")
except KeyboardInterrupt:
    print("\nExiting chat, Bye!")