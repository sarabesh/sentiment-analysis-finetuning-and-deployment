from fastapi import FastAPI
from pydantic import BaseModel
from .backend import predict

app = FastAPI()

class Input(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "Sentiment Analysis Inference API is up and running. Use a POST request at /classify to analyze sentiment."}


@app.post("/classify")
async def generate(input_: Input):
    text = input_.text
    prediction, confidence_prob = predict(text)
    return {"predicted class": prediction, "confidence": confidence_prob}
