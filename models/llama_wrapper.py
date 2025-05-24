from llama_cpp import Llama

class LlamaChat:
    def __init__(self, model_path, n_gpu_layers=24, n_threads=12, n_ctx=4096):
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_ctx=n_ctx
        )

    def generate(self, prompt, max_tokens=200):
        output = self.llm(prompt, max_tokens=max_tokens, stop=["<|user|>", "<|assistant|>", "<|system|>"])
        return output["choices"][0]["text"].strip()


