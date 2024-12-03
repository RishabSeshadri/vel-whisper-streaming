from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto")

# Create FastAPI app
app = FastAPI()

# Define a request model
class Request(BaseModel):
    prompt: str
    max_length: int = 100

# Define the generate route
@app.post("/generate")
async def generate_text(request: Request):
    # Tokenize the input prompt
    inputs = tokenizer(request.prompt, return_tensors="pt")
    
    # Move the input_ids tensor to the same device as the model
    input_ids = inputs["input_ids"].to(model.device)

    # Generate text from the model
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=request.max_length)

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
