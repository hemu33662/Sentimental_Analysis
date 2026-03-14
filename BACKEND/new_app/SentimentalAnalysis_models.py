import os
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FRONDEND_DIR = os.path.join(ROOT_DIR, "FRONTEND")
# BACKEND_DIR = os.path.join(ROOT_DIR, "BACKEND")
LOADS_DIR = os.path.join(ROOT_DIR, "Sentimental_Analysis_Loads")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("########### model ##########")

print("#####################")
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


def predict_sentiment(text, algo):
    text_vectorized = preprocess_text(text)

    if algo == "SVM":
        prediction = svm_model.predict(text_vectorized)
    elif algo == "SVM_Pipeline":
        prediction = svc_pipeline_model.predict(
            [text]
        )
    elif algo == "NaiveBayes":
        prediction = naivebayes_model.predict(text_vectorized)
    elif algo == "MultinomialNB":
        prediction = multinomialnb_model.predict(
            [text]
        )
    else:
        return "Invalid Algorithm Selected"

    return "Positive" if prediction[0] == 1 else "Negative"
