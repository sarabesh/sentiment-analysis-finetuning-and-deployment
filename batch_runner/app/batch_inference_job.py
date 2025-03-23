
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import pandas as pd
import requests
from datetime import datetime  


def send_slack_alert(message):
    print(f"Sending Slack alert: {message}")
    # webhook_url = os.getenv("SLACK_WEBHOOK_URL")  #load as env from kubernetes secrets
    webhook_url = "https://hooks.slack.com/services/**use**your**webhook**url**here**"
    payload = {"text": message}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(webhook_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("Slack alert sent successfully!")
    else:
        print(f"Failed to send Slack alert: {response.text}")

def check_data_distribution(df):
    # Calculate class distribution
    class_dist = df['predicted_class'].value_counts()
    
    # Calculate percentages
    total = len(df)
    class_0_pct = (class_dist.get(0, 0) / total) * 100
    class_1_pct = (class_dist.get(1, 0) / total) * 100
    print(f"Class 0: {class_0_pct:.2f}%")
    print(f"Class 1: {class_1_pct:.2f}%")
    # Check for major imbalance (threshold set at 70-30)
    imbalance_threshold = 70
    
    if class_0_pct >= imbalance_threshold or class_1_pct >= imbalance_threshold:
        alert_message = f"""
        WARNING: Major class imbalance detected!
        Class 0: {class_0_pct:.2f}%
        Class 1: {class_1_pct:.2f}%
        """
        send_slack_alert(alert_message)
        return False, alert_message
    send_slack_alert("Class distribution is balanced")
    return True, "Class distribution is balanced"


@torch.no_grad()
def predict_batch(data,output_path, batch_size=1000):
    
    model.to(device)
    model.eval()
    texts = data["text"].tolist()
    predictions = []
       
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process in batches
    for i in range(0, len(texts), batch_size):
        print(f"Processing batch {i // batch_size + 1} of {len(texts) // batch_size + 1}")
        batch_texts = texts[i : i + batch_size]
        print(f"Batch size: {len(batch_texts)}")
        # Tokenize batch and move to GPU
        tokens = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt").to(device)
        
        # Run inference
        outputs = model(**tokens)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # Get probabilities

        # Get predicted labels & confidence scores
        pred_labels = torch.argmax(probs, dim=1).cpu().numpy()  # Move back to CPU for processing
        predictions.extend(pred_labels)
           # Create new dictionary with batch texts and predicted labels
        batch_results = {"text": batch_texts, "predicted_class": pred_labels}
        
        # Convert dictionary to DataFrame
        batch_df = pd.DataFrame(batch_results)

        # Append batch predictions to CSV
        header = not os.path.exists(output_path)  # Write header only if file doesn't exist
        batch_df.to_csv(output_path, mode="a", index=False, header=header)
        print(f"Saved batch {i // batch_size + 1} to {output_path}")
      
    return predictions


def fetch_inputs(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records from {input_path}")
    return df


print("Started batch_inference")
batch_inputs = fetch_inputs("/mnt/minikube/data/new_texts.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#currently adding model as part of docker image. It would be better to use a model registry like MLflow to store and load the model.
model = AutoModelForSequenceClassification.from_pretrained(os.path.abspath('./app/models/finetuned-bert'))
model.to(device)
print("Loaded tokenizer and model")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"/mnt/minikube/predictions/test_results_{timestamp}.csv"

results = predict_batch(batch_inputs,output_path=filename)
batch_inputs['predicted_class'] = results
print("Predictions made")
check_data_distribution(batch_inputs)
print("Data distribution checked")
