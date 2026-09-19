import ollama

response = ollama.chat(
    model="qwen2.5:0.5b",
    messages=[
        {
            "role": "user",
            "content": "Explain software testing in 2 simple points."
        }
    ]
)

print(response["message"]["content"])