import os
from transformers import pipeline

def test_my_ai():
    # 1. Point to your personalized model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "Sentimental_Analysis_Loads", "custom_llm_model")

    if not os.path.exists(model_path):
        print("❌ Error: You haven't trained your model yet!")
        print("Please run `python train_custom_llm.py` first.")
        return

    print("🤖 Loading your very own RoBERTa Custom LLM...")
    
    # 2. Load the model using the Hugging Face pipeline
    classifier = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

    # 3. Create some tricky test reviews
    reviews = [
        "The salmon was absolutely perfect, melts in your mouth!", # Positive
        "Honestly, the fries tasted like old rubber.", # Negative
        "It wasn't terrible, but it certainly wasn't good for $40.", # Negative
        "I was completely blown away by the attentive service." # Positive
    ]

    print("\n--- 🧪 TESTING YOUR AI ---")
    for review in reviews:
        prediction = classifier(review)[0]
        # LABEL_1 is the default for Positive in SequenceClassification
        sentiment = "Positive" if prediction['label'] in ['LABEL_1', 'POSITIVE', '1'] else "Negative"
        confidence = round(prediction['score'] * 100, 2)
        
        icon = "🟢" if sentiment == "Positive" else "🔴"
        print(f"{icon} Review: '{review}'")
        print(f"   => Label: {sentiment} (Confidence: {confidence}%)\n")

if __name__ == "__main__":
    test_my_ai()
