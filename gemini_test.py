from google import genai
import os

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Explain software testing in 2 simple points."
)

print(interaction.output_text)