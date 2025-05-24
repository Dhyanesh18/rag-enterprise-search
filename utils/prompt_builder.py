def build_prompt(system_prompt, context_list, user_input):
    prompt = f"<|system|>\n{system_prompt}\n"
    for ctx in context_list:
        prompt += f"<|user|>\n{ctx['user']}\n<|assistant|>\n{ctx['assistant']}\n\n"
    prompt += f"<|user|>\n{user_input}\n<|assistant|>\n"
    return prompt
