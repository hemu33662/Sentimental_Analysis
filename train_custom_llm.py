import pandas as pd
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "DATASET", "Restaurant_Reviews.tsv")
SAVE_DIR = os.path.join(BASE_DIR, "Sentimental_Analysis_Loads", "custom_llm_model")

def main():
    print("🚀 Loading dataset...")
    # Read the TSV dataset
    df = pd.read_csv(DATA_PATH, sep='\t')
    
    # Rename columns for the transformers library
    # 'Review' -> 'text', 'Liked' -> 'labels'
    df = df.rename(columns={"Review": "text", "Liked": "labels"})
    
    # Convert pandas dataframe to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    # Split the dataset into train and test sets (80% train, 20% test)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    print("🔄 Loading Tokenizer & Model (RoBERTa - The Better Model)...")
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We have 2 labels: 0 (Negative) and 1 (Positive)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("📝 Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,              # 3 Passes over the data
        per_device_train_batch_size=8,   # Batch size for training
        per_device_eval_batch_size=8,    # Batch size for evaluation
        warmup_steps=100,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Strength of weight decay
        logging_steps=10,
        eval_strategy="epoch",           # Evaluate at the end of every epoch (renamed in modern transformers)
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )
    
    print("🔥 Starting Training! (This might take a few minutes depending on your PC)...")
    trainer.train()
    
    print(f"✅ Training Complete! Saving your custom LLM to: {SAVE_DIR}")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    
    print("\n🎉 Custom LLM is saved! You can now use it in your backend API.")

if __name__ == "__main__":
    main()
