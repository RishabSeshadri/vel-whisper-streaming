from transformers import LlamaTokenizer
import httpx

# Define the API endpoint and model
API_URL = "http://localhost:8000/generate"
MODEL_NAME = "llama-3.2:1b"

# Define the input prompt
prompt = "Hello! How are you?"

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Tokenize the input prompt
tokenized_prompt = tokenizer(prompt, return_tensors="pt")

# Prepare the request payload
payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "options": {
        "max_tokens": 50,
        "temperature": 0.7
    },
    "pad_token_id": tokenizer.pad_token_id,
    "attention_mask": tokenized_prompt['attention_mask'].tolist()
}

# Make a POST request to the API with extended timeout
try:
    response = httpx.post(API_URL, json=payload, timeout=60.0)
    response.raise_for_status()

    # Parse and print the response
    result = response.json()
    print("Llama 3.2 Response:", result.get("generated_text", "No response text found"))
except httpx.TimeoutException:
    print("The request timed out. The model might be taking too long to respond.")
except httpx.HTTPError as exc:
    print(f"HTTP Error occurred: {exc}")
except Exception as e:
    print(f"An error occurred: {e}")