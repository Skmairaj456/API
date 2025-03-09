from fastapi import FastAPI, WebSocket
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

app = FastAPI()

# Load AI model (Default: GPT-2 for easy deployment)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")  # Change this when needed

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    ai_model = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    ai_model = None  # Prevent crashes

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        user_input = await websocket.receive_text()
        
        if ai_model:
            response = ai_model(user_input, max_length=200)[0]['generated_text']
        else:
            response = "Error: AI model failed to load."
        
        await websocket.send_text(response)

# Run this with: uvicorn main:app --host 0.0.0.0 --port 8000
