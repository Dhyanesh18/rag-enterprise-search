def build_prompt(system_prompt, context_list, user_input):
    """
    Constructs a multi-turn prompt for an LLM using a system prompt, retrieved context chunks,
    and a user question. Formats the input using role tags like <|system|>, <|user|>, <|assistant|>.

    Args:
        system_prompt (str): Instructions or role definition for the assistant.
        context_list (list[dict]): A list of context chunks. Each chunk is a dict with:
            - 'content' (str): The actual text.
            - 'score' (float, optional): Relevance score (lower = more relevant).
            - 'meta' (dict, optional): Metadata such as chunk index and source file.
        user_input (str): The user’s question.

    Returns:
        str: A formatted prompt ready to be fed to the LLM.
    """
    # Start the prompt with the system instructions
    prompt = f"<|system|>\n{system_prompt.strip()}\n"

    # Build the context section using the top-k retrieved chunks
    context_section = ""
    for i, ctx in enumerate(context_list):
        score_str = f" (Score: {ctx['score']:.4f})" if ctx.get("score") is not None else ""
        source_info = ""
        if ctx.get("meta"):
            chunk_idx = ctx["meta"].get("chunk_index", "N/A")
            source = ctx["meta"].get("source", "unknown").split("/")[-1]
            source_info = f"From {source}, chunk {chunk_idx}"
        context_section += f"Context {i+1}{score_str} {source_info}:\n{ctx['content'].strip()}\n\n"

    # Inject context into the <|user|> section
    prompt += f"<|user|>\nHere are some relevant context pieces:\n{context_section.strip()}\n"

    # Add the user's actual question
    prompt += f"\nQuestion: {user_input.strip()}\n"

    # Add answering instructions and constraints for the LLM
    prompt += """- If the context is not relevant, respond: "I don't know based on the provided context"
- Context 1 IS THE MOST RELEVANT, Context 2 IS THE SECOND MOST RELEVENT, and so on (BETTER SCORE MEANS BETTER RELEVANCE).
- Provide all the relevant information from the context in your answer.
- INCLUDE ALL THE IMPORTANT DETAILS LIKE EDGE CASES, VERSIONS, SPECIAL CASES, REQUIREMENTS, BOTTLENECKS."""

    prompt += "<|assistant|>\n"
    return prompt
