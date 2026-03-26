import os
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FRONDEND_DIR = os.path.join(ROOT_DIR, "FRONTEND")
# BACKEND_DIR = os.path.join(ROOT_DIR, "BACKEND")
LOADS_DIR = os.path.join(ROOT_DIR, "Sentimental_Analysis_Loads")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Avoid noisy prints in production logs.
svm_model = joblib.load(
    os.path.join(LOADS_DIR, "Sentimental_Analysis_svm_model.pkl")
)
svc_pipeline_model = joblib.load(
    os.path.join(LOADS_DIR, "Sentimental_Analysis_SVM_pipeline.pkl")
)
naivebayes_model = joblib.load(
    os.path.join(LOADS_DIR, "Sentimental_Analysis_naive_bayes.pkl")
)
multinomialnb_model = joblib.load(
    os.path.join(LOADS_DIR, "Sentimental_Analysis_MultinomialNB_model.pkl")
)

tfidf_vect = joblib.load(
    os.path.join(LOADS_DIR, "Sentimental_Analysis_tfidf.pkl")
)


def preprocess_text(text):
    return tfidf_vect.transform([text])


def _prediction_to_sentiment(prediction) -> str:
    """
    Convert various model output formats into a stable 'Positive'/'Negative' string.
    """
    try:
        first = prediction[0]
    except Exception:
        first = prediction

    first_str = str(first).strip().lower()

    # Common numeric + label formats across scikit + custom models
    if first in (1, "1") or first_str in {"label_1", "positive"}:
        return "Positive"
    if first in (0, "0") or first_str in {"label_0", "negative"}:
        return "Negative"

    # Fallback: treat unknown labels as Negative (safer than crashing)
    return "Negative"


def predict_sentiment(text, algo):
    if algo == "Custom_LLM":
        try:
            custom_model_dir = os.path.join(LOADS_DIR, "custom_llm_model")
            
            # 1. OPTION A: Local Development (Uses PyTorch if folder exists)
            if os.path.exists(custom_model_dir):
                from transformers import pipeline
                classifier = pipeline("sentiment-analysis", model=custom_model_dir, tokenizer=custom_model_dir)
                prediction = classifier(text)[0]
                return "Positive" if prediction['label'] in ['LABEL_1', 'POSITIVE', '1'] else "Negative"
            
            # 2. OPTION B: Production Deployment on Render (Uses Hugging Face Free API)
            else:
                import requests
                # Replace with YOUR Hugging Face model URL once you upload it!
                # E.g., https://api-inference.huggingface.co/models/hemu33662/restaurant-sentiment-model
                default_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
                API_URL = os.environ.get("HF_MODEL_URL", default_url)
                
                headers = {}
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"
                
                # Add a timeout so external calls can't hang indefinitely.
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json={"inputs": text},
                    timeout=15,
                )
                
                if response.status_code != 200:
                    return "API Error"
                
                # Hugging Face returns a list of dictionaries. Sort or find the highest score.
                try:
                    predictions = response.json()[0]
                    best_pred = max(predictions, key=lambda x: x["score"])
                except Exception:
                    return "API Error"
                
                # We assume the user's custom model outputs 'LABEL_1' (positive) or 'LABEL_0' (negative), 
                # or 'positive'/'negative'
                label = str(best_pred.get("label", "")).lower()
                return "Positive" if label in ["label_1", "positive", "1"] else "Negative"

        except Exception:
            # Never leak exception details to clients.
            return "Custom LLM Error"

    try:
        text_vectorized = preprocess_text(text)

        if algo == "SVM":
            prediction = svm_model.predict(text_vectorized)
        elif algo == "SVM_Pipeline":
            prediction = svc_pipeline_model.predict([text])
        elif algo == "NaiveBayes":
            prediction = naivebayes_model.predict(text_vectorized)
        elif algo == "MultinomialNB":
            # Keep consistent with other vectorizer-based models.
            # If the saved model expects raw text, fall back to raw input.
            try:
                prediction = multinomialnb_model.predict(text_vectorized)
            except Exception:
                prediction = multinomialnb_model.predict([text])
        else:
            return "Invalid Algorithm Selected"

        return _prediction_to_sentiment(prediction)
    except Exception:
        # Prevent `/output` from returning 500 due to model exceptions.
        return "Model Error"
