# Fine-Tuning BERT for Sentiment Analysis and Deployment

## Overview
This repo simulates how an ML model moves to production in an industry setting. The goal is to build, deploy, monitor, and retrain a sentiment analysis model using Kubernetes (minikube) and FastAPI.

## Note
Models and dataset files are not included in this repository due to size constraints. You can find the dataset at: https://www.kaggle.com/datasets/kazanova/sentiment140

## Tasks & Implementation

### Running a Basic Sentiment Analysis
- **Dataset**: Publicly available labeled dataset containing tweets with sentiment labels. `https://www.kaggle.com/datasets/kazanova/sentiment140`
- **Notebook**: `bert-finetuning-sentiment-analysis-training.ipynb`
  - Performed initial dataset analysis and basic cleaning.
  - Used a Hugging Face BERT model for fine-tuning on ~60% of the dataset.
  - Saved trained model artifacts for future use.
  - Notebook was run on Kaggle, and model files were recovered from persisted files.
- **Location**: `dataset/`

### Wrapping the Model as an API
- **Folder**: `inference/`
- **Framework**: FastAPI
- **Functionality**:
  - API accepts a tweet as input and returns sentiment prediction along with confidence scores. It run at endpoint '/predict' as post request and return predicted class and confidence score.
- **Deployment**:
  - Model is included as part of `inference/app`.
  - Built and deployed in local minikube.
  - Tested using Postman (POST API call).
- **Files**:
  - `app\`:
    - `models`
    - `main.py`
  - `Dockerfile`
  - `inference-deployment.yaml`

### Asynchronous Prediction for Large Batches
- **Folder**: `batch_runner/`
- **Functionality**:
  - Periodic job in Kubernetes (minikube) runs daily to predict sentiment on a new batch of data.
  - Uses half of the remaining dataset to simulate incoming data.
  - Saves results to a mounted volume in minikube.
- **Deployment**:
  - Requires `minikube-volume` to be mounted.
  - Contains both a Kubernetes **Job** (for testing) and a **CronJob** (set to run daily).
- **Files**:
  - `app/`
    - `models`
    - `batch_runner.py`
  - `Dockerfile`
  - `batch_job_manual.yaml` (for testing)
  - `batch_job.yaml` (for daily execution)

### Data Drift & Model Retraining
#### Data Drift Monitoring
- `check_data_distribution()` Function added to `batch_runner.py` to check class distribution in predicted data.
- Sends Slack alerts in case of a major class imbalance.
- **Note**: Slack webhook URL is hardcoded; ideally, use Kubernetes secrets.

#### Model Retraining
- **Folder**: `retrain_runner/`
- Retrains the model periodically using additional data.
- Saves new model artifacts.
- Requires `minikube-volume` for accessing fine-tuning data.
- Tested with limited data.
- **Deployment**:
  - **Job** (for testing)
  - **CronJob** (set to run once a month)
- **Files**:
  - `Dockerfile`
  - `retrain_job.yaml`
  - `retrain_cronjob.yaml`

## Further steps

- Ensure `minikube-volume` is mounted properly to access batch and retraining data, or use other methods for data access.
- Implement a more robust data drift monitoring system.
- Consider using Kubernetes secrets for sensitive information (e.g., Slack webhook).

---
