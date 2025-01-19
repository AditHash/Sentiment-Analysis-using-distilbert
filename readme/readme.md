# Sentiment Analysis using DistilBERT

This repository contains a complete pipeline for performing sentiment analysis using a fine-tuned DistilBERT model. The dataset contains reviews, which are processed and labeled to identify sentiments as **Negative**, **Neutral**, or **Positive**. The project leverages the Hugging Face Transformers library and PyTorch for fine-tuning the model.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Pipeline Description](#pipeline-description)
6. [Evaluation and Results](#evaluation-and-results)
7. [Example Prediction](#example-prediction)
8. [License](#license)

---

## Project Overview

The goal of this project is to classify text reviews into three sentiment categories:

- **Negative**: Ratings 1-2
- **Neutral**: Rating 3
- **Positive**: Ratings 4-5

A pre-trained DistilBERT model is fine-tuned for this classification task, ensuring high accuracy and efficiency.

---

## Features

- Text preprocessing (cleaning, tokenization).
- Custom sentiment label assignment based on ratings.
- Fine-tuning of DistilBERT for sentiment classification.
- Evaluation metrics including accuracy, precision, recall, and F1 score.
- Example usage for predicting sentiment on custom inputs.

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Sentiment-Analysis-using-distilbert.git
   cd Sentiment-Analysis-using-distilbert
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Dependencies:
   - `pandas`
   - `transformers`
   - `torch`
   - `datasets`
   - `scikit-learn`

3. Disable Weights & Biases logging (optional):
   ```bash
   export WANDB_DISABLED=true
   ```

---

## Usage

1. **Preprocess Data**: Prepare the data by cleaning and assigning sentiment labels.
2. **Train the Model**: Fine-tune the DistilBERT model using the provided training pipeline.
3. **Evaluate the Model**: Evaluate performance on a test dataset.
4. **Predict Sentiments**: Use the trained model to predict the sentiment of custom text inputs.

---

## Pipeline Description

### 1. Data Preprocessing

- Assign sentiment labels based on the rating column:
  - Ratings 1-2 → Negative (Label: 0)
  - Rating 3 → Neutral (Label: 1)
  - Ratings 4-5 → Positive (Label: 2)
- Clean and preprocess text:
  - Remove special characters.
  - Convert text to lowercase.

### 2. Tokenization and Dataset Preparation

- Use `AutoTokenizer` from Hugging Face for tokenization.
- Pad and truncate sequences to a maximum length of 512 tokens.

### 3. Model Training

- Fine-tune a pre-trained `distilbert-base-uncased` model for sentiment classification.
- Training parameters:
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 3
  - Weight decay: 0.01

### 4. Evaluation

- Metrics: Accuracy, Precision, Recall, F1 score.

### 5. Inference

- Predict sentiment on custom inputs using the trained model.

---

## Evaluation and Results

- The fine-tuned model achieves high accuracy and robust performance across all sentiment categories.
- Example metrics (from training logs):
  - **Accuracy**: ~90%
  - **F1 Score**: ~0.89
  - **Precision**: ~0.90
  - **Recall**: ~0.88

---

## Example Prediction

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./results")  # Path to fine-tuned model

# Custom text input
text = "The product quality was excellent, but the delivery took too long."

# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
print(f"Predicted Sentiment: {sentiment_map[predicted_class]}")
```

---

## License

This project is licensed under the MIT License. Feel free to use and modify the code as per your needs.

---

Happy sentiment analysis! 🚀

