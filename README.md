# 🤖 DistilBERT Question Answering on Amazon SageMaker

This project demonstrates how to deploy DistilBERT on AWS SageMaker to build a scalable Question-Answering (QA) API, using the Hugging Face deployment integration.

distilbert-base-uncased-distilled-squad is a lightweight, efficient QA model trained on the SQuAD dataset, enabling real-time extraction of answers from text.

# 📌 Project Overview
Configures Hugging Face model for SageMaker

Creates a SageMaker endpoint with DistilBERT

Sends question + context to model for inference

Returns the predicted answer

Optionally deletes endpoint to avoid costs

# 🎯 Goal

To deploy a serverless NLP inference endpoint for Question-Answering using Amazon SageMaker.

# 🧠 Model Used
Attribute	Value
Model	distilbert-base-uncased-distilled-squad
Task	question-answering
Framework	Hugging Face Transformers
Cloud Service	Amazon SageMaker

DistilBERT = a distilled (compressed) version of BERT
SQuAD = Stanford Question Answering Dataset

# 🛠️ Prerequisites

AWS Account

IAM role with SageMaker permissions

Python 3.x

HuggingFace + SageMaker SDK

# 📦 Installation
pip install sagemaker -U
pip install transformers

# 🚀 Deployment Steps
### 1️⃣ Define Hugging Face model config
hub = {
  'HF_MODEL_ID':'distilbert-base-uncased-distilled-squad',
  'HF_TASK':'question-answering'
}

### 2️⃣ Create SageMaker HuggingFaceModel
from sagemaker.huggingface.model import HuggingFaceModel

huggingface_model = HuggingFaceModel(
    env=hub,
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version='py39'
)

### 3️⃣ Deploy endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)

🧪 Inference Example
data = {
    "inputs": {
        "question": "Where does Raviteja work?",
        "context": "Raviteja is doing a Data Science internship at Cognifyz Technologies."
    }
}

predictor.predict(data)

### ✅ Output Example
{
  "answer": "Cognifyz Technologies",
  "score": 0.98
}

# 🧹 Cleanup
predictor.delete_endpoint()

To avoid charges, always delete the endpoint after testing

# 📈 Performance Notes
Feature	Benefit
Distilled BERT	Faster than BERT with minimal accuracy loss
SageMaker Endpoint	Scalable inference deployment
HF Integration	Plug-and-play model serving

# 🌟 Future Enhancements

Switch to SageMaker Serverless for cost-efficiency

Add API Gateway + Lambda front-end

Streamlit UI for QA chatbot interface

Fine-tune custom QA dataset

Add evaluation metrics (F1, EM)

# 🏁 Conclusion

This project deploys a production-ready NLP QA model on AWS SageMaker using Hugging Face, demonstrating a fast and scalable approach to cloud-based inference for real-time question answering.
