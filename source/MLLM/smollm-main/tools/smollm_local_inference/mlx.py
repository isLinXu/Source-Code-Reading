from mlx_lm import load, generate

model, tokenizer = load("HuggingFaceTB/SmolLM2-1.7B-Instruct-Q8-mlx")

prompt = "Hello"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(response)
