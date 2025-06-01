def run_standard_pipeline(query, pipeline):
    """Run a standard retrieval pipeline with the given query and pipeline configuration."""
    return pipeline.retrieve(query)

def run_hyde_pipeline(query, llm, pipeline):
    """Run a HYDE pipeline with the given query, LLM, and retrieval pipeline."""
    prompt = f"""Write a short, factual passage that would answer this question:
    
Question: {query}

Passage:"""
    hypothetical_passage = llm.generate(prompt)
    print(f"Hypothetical Passage: {hypothetical_passage}")
    return pipeline.retrieve(hypothetical_passage)
