from models.llama_wrapper import LlamaChat
from memory.embedder import Embedder
from memory.memory_store import MemoryStore
from memory.logger import JSONLogger
from utils.prompt_builder import build_prompt

system_prompt = "You are a helpful AI assistant. Answer concisely and clearly."

llama = LlamaChat(model_path="./models/openhermes-2.5-mistral-7b.Q4_K_M.gguf")
embedder = Embedder()
memory = MemoryStore()
logger = JSONLogger()

print("Chat with Mistral (type Ctrl+C to stop, or type /reset to clear memory)")

try:
    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "/reset":
            memory.reset()
            print("Memory cleared!")
            continue

        query_embed = embedder.get_embedding(user_input)
        relevant = memory.retrieve(query_embed)

        prompt = build_prompt(system_prompt, relevant, user_input)
        response = llama.generate(prompt)

        print(f"Assistant: {response}\n")

        logger.log(user_input, response)

        memory.store(user_input, response, query_embed)


except KeyboardInterrupt:
    print("\nExiting chat. Bye!")
