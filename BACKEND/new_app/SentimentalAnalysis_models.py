import os
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Allow overriding the models directory via an environment variable so it works on Render
LOADS_DIR = Path(
    os.environ.get(
        "LOADS_DIR",
        ROOT_DIR / "Sentimental_Analysis_Loads",
    )
)


def _load_model(filename):
    model_path = LOADS_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Ensure the Sentimental_Analysis model .pkl files are deployed and "
            "that LOADS_DIR (if set) points to the correct directory."
        )
    return joblib.load(model_path)


print("########### model ##########")
print(f"Using LOADS_DIR={LOADS_DIR}")

svm_model = _load_model("Sentimental_Analysis_svm_model.pkl")
svc_pipeline_model = _load_model("Sentimental_Analysis_SVM_pipeline.pkl")
naivebayes_model = _load_model("Sentimental_Analysis_naive_bayes.pkl")
multinomialnb_model = _load_model("Sentimental_Analysis_MultinomialNB_model.pkl")
tfidf_vect = _load_model("Sentimental_Analysis_tfidf.pkl")


def preprocess_text(text):
    return tfidf_vect.transform([text])


def predict_sentiment(text, algo):
    text_vectorized = preprocess_text(text)

    if algo == "SVM":
        prediction = svm_model.predict(text_vectorized)
    elif algo == "SVM_Pipeline":
        prediction = svc_pipeline_model.predict([text])
    elif algo == "NaiveBayes":
        prediction = naivebayes_model.predict(text_vectorized)
    elif algo == "MultinomialNB":
        prediction = multinomialnb_model.predict([text])
    else:
        return "Invalid Algorithm Selected"

    return "Positive" if prediction[0] == 1 else "Negative"
