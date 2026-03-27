import os
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FRONDEND_DIR = os.path.join(ROOT_DIR, "FRONTEND")
# BACKEND_DIR = os.path.join(ROOT_DIR, "BACKEND")
LOADS_DIR = os.path.join(ROOT_DIR, "Sentimental_Analysis_Loads")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global cache for small models to avoid reloading if memory allows, 
# but only load them inside the function to keep start memory low.
_models_cache = {}

import time
import logging

def get_model(algo):
    """
    On-demand loading and caching of models.
    """
    if algo not in _models_cache:
        start_load = time.time()
        
        # Handle Custom LLM (Transformers)
        if algo == "Custom_LLM":
            from transformers import pipeline
            custom_model_dir = os.path.join(LOADS_DIR, "custom_llm_model")
            if os.path.exists(custom_model_dir):
                _models_cache[algo] = pipeline("sentiment-analysis", model=custom_model_dir, tokenizer=custom_model_dir)
            else:
                return None
        
        # Handle Scikit-learn models (Joblib)
        else:
            model_files = {
                "SVM": "Sentimental_Analysis_svm_model.pkl",
                "SVM_Pipeline": "Sentimental_Analysis_SVM_pipeline.pkl",
                "NaiveBayes": "Sentimental_Analysis_naive_bayes.pkl",
                "MultinomialNB": "Sentimental_Analysis_MultinomialNB_model.pkl",
                "tfidf": "Sentimental_Analysis_tfidf.pkl"
            }
            if algo not in model_files and algo != "tfidf":
                return None
                
            file_path = os.path.join(LOADS_DIR, model_files.get(algo, "Sentimental_Analysis_tfidf.pkl"))
            _models_cache[algo] = joblib.load(file_path)
        
        load_time = (time.time() - start_load) * 1000
        logging.info(f"Loaded {algo} model in {load_time:.2f}ms")
        
    return _models_cache[algo]

def preprocess_text(text):
    tfidf_vect = get_model("tfidf")
    return tfidf_vect.transform([text])


def _prediction_to_sentiment(prediction) -> str:
    try:
        if hasattr(prediction, "__getitem__"):
            first = prediction[0]
        else:
            first = prediction
    except Exception:
        first = prediction

    first_str = str(first).strip().lower()
    if first in (1, "1") or first_str in {"label_1", "positive", "1", "joy"}:
        return "Positive"
    return "Negative"


def predict_sentiment(text, algo):
    if not text:
        return "Empty Input"

    start_total = time.time()
    result = "Error"

    if algo == "Custom_LLM":
        try:
            is_render = os.environ.get("RENDER") == "true"
            
            # Use local model ONLY if we are NOT on Render
            if not is_render:
                classifier = get_model("Custom_LLM")
                if classifier:
                    prediction = classifier(text)[0]
                    label = str(prediction['label']).upper()
                    result = "Positive" if label in ['LABEL_1', 'POSITIVE', '1'] else "Negative"
                else:
                    # Fallback to API if local directory missing
                    is_render = True
            
            if is_render:
                import requests
                API_URL = os.environ.get("HF_MODEL_URL", "https://router.huggingface.co/hf-inference/models/HemanthNasaram/restaurant-sentiment-model")
                headers = {}
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"
                
                response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=15)
                
                if response.status_code == 200:
                    json_data = response.json()
                    # HF API often returns 'estimated_time' if loading
                    if isinstance(json_data, dict) and "estimated_time" in json_data:
                        result = f"API loading... ({int(json_data['estimated_time'])}s)"
                    else:
                        predictions = json_data[0]
                        best_pred = max(predictions, key=lambda x: x["score"])
                        label = str(best_pred.get("label", "")).lower()
                        result = "Positive" if label in ["label_1", "positive", "1", "joy"] else "Negative"
                else:
                    result = f"API Error {response.status_code}"

        except Exception as e:
            logging.error(f"Custom LLM Error: {str(e)}")
            result = "Custom LLM Error"
    else:
        try:
            if algo == "SVM":
                prediction = get_model("SVM").predict(preprocess_text(text))
            elif algo == "SVM_Pipeline":
                prediction = get_model("SVM_Pipeline").predict([text])
            elif algo == "NaiveBayes":
                prediction = get_model("NaiveBayes").predict(preprocess_text(text))
            elif algo == "MultinomialNB":
                model = get_model("MultinomialNB")
                try:
                    prediction = model.predict(preprocess_text(text))
                except Exception:
                    prediction = model.predict([text])
            else:
                return "Invalid Algorithm"
            
            result = _prediction_to_sentiment(prediction)
        except Exception as e:
            logging.error(f"Model error ({algo}): {str(e)}")
            result = "Model Error"

    total_time = (time.time() - start_total) * 1000
    logging.info(f"Total prediction time ({algo}): {total_time:.2f}ms")
    return result
