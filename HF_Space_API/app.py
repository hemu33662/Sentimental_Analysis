from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load your custom model
classifier = pipeline("sentiment-analysis", model="HemanthNasaram/restaurant-sentiment-roberta")

class SentimentRequest(BaseModel):
    inputs: str

@app.post("/")
async def analyze_sentiment(request: SentimentRequest):
    # We use request.inputs to replicate the exact format your Django app was already sending
    output = classifier(request.inputs)
    return output
