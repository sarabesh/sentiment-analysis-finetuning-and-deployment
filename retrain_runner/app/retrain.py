#modified from training notebook.
#Objective: retrain the model with the new data and save the new model version.

import torch
import os
import pandas as pd
from data_processes import clean_data
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from datetime import datetime


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#helper functions
def fetch_inputs(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.df = pd.read_csv(input_path, nrows=50000, encoding="ISO-8859-1", header=None , names=['target', 'ids', 'date', 'flag' , 'user' , 'text' ])

    print(f"Loaded {len(df)} records from {input_path}")
    return df

# Tokenize text
def tokenize_function(examples):
    return tokenizer(examples["text"],truncation=True, padding="max_length", max_length=128)
df = fetch_inputs("/mnt/minikube/data/training.csv")

#removing unnecessary columns for training, and finxing the target column
print("Cleaning data...")
clean_df = clean_data(df)

train_data = Dataset.from_dict({
    'text': clean_df['text'].tolist(),
    'target': clean_df['target'].tolist()
})

# Load the model
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#currently adding model as part of docker image. It would be better to use a model registry like MLflow to store and load the model.
model = AutoModelForSequenceClassification.from_pretrained(os.path.abspath('./models/finetuned-bert'))
model.to(device)

train_data = train_data.map(tokenize_function, batched=True)
# Remove original text column to keep only model inputs
train_data = train_data.remove_columns(["text"])

# Ensure labels are in correct format
train_data = train_data.map(lambda x: {"labels": torch.tensor(int(x["target"]))})

# Set dataset format for PyTorch
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

#freeze weight of the model for fine-tuning
# freeze all base model parameters
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# unfreeze base model pooling layers
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# hyperparameters
batch_size = 64
num_epochs = 10
lr = 2e-4


# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f"/mnt/minikube/finetuned_bert/model_{timestamp}"

training_args = TrainingArguments(
    output_dir=model_dir,
    save_strategy="epoch",
    report_to='none',
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    compute_metrics=compute_metrics,
)

print("All loaded, Starting training...")
#training
trainer.train()


# Save the fine-tuned model and tokenizer
model.save_pretrained(model_dir+"/model")
tokenizer.save_pretrained(model_dir+"/tokenizer")

print(f"Model saved in directory: {model_dir}")