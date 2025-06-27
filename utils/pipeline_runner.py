def run_standard_pipeline(query, pipeline):
    """
    Run a standard retrieval pipeline using the original query.

    Args:
        query (str): The user question or input query.
        pipeline: A retrieval pipeline object that has a `.retrieve(query)` method.

    Returns:
        list[dict]: Retrieved documents/passages relevant to the query.
    """
    return pipeline.retrieve(query)

def run_hyde_pipeline(query, llm, pipeline):
    """
    Run a HYDE (Hypothetical Document Embeddings) pipeline:
    - First generate a hypothetical answer passage from the query using an LLM.
    - Then retrieve documents using that generated passage.

    Args:
        query (str): The original user query.
        llm: A language model object with a `.generate(prompt)` method (e.g., Mistral wrapper).
        pipeline: A retrieval pipeline with a `.retrieve(query)` method, typically accepting text.

    Returns:
        list[dict]: Retrieved documents/passages based on the hypothetical passage.

    Notes:
        This method can improve retrieval by using LLM-generated context as the query,
        especially for vague or underspecified original questions.
    """
    prompt = f"""Write a short, factual passage that would answer this question:
    
Question: {query}

Passage:"""
    hypothetical_passage = llm.generate(prompt)
    print(f"Hypothetical Passage: {hypothetical_passage}")
    return pipeline.retrieve(hypothetical_passage)
