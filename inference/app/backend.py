import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#currently adding model as part of docker image. It would be better to use a model registry like MLflow to store and load the model.
model = AutoModelForSequenceClassification.from_pretrained(os.path.abspath('./app/models/finetuned-bert'))
model.to(device)


@torch.no_grad()
def predict(text):
    tokens = tokenizer(text, truncation=True, return_tensors="pt").to(device)
    output = model(**tokens)
    logits = output["logits"].detach()
    probs = logits.softmax(dim=1)[0]
    predicted_class = probs.argmax().item()
    return predicted_class, probs[predicted_class].item()
