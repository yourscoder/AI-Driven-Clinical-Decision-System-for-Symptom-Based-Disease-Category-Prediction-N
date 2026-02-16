import pandas as pd
import numpy as np
import torch
import spacy
import re

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])



DISEASE_TO_CATEGORY = {
    "Common Cold":"Respiratory","Influenza":"Respiratory","COVID-19":"Respiratory",
    "Pneumonia":"Respiratory","Tuberculosis":"Respiratory","Asthma":"Respiratory",
    "Bronchitis":"Respiratory","Sinusitis":"Respiratory",

    "Heart Disease":"Cardiovascular","Hypertension":"Cardiovascular","Stroke":"Cardiovascular",

    "Gastritis":"Gastrointestinal","Food Poisoning":"Gastrointestinal",
    "Irritable Bowel Syndrome (IBS)":"Gastrointestinal","Ulcer":"Gastrointestinal",
    "Liver Disease":"Gastrointestinal",

    "Migraine":"Neurological","Epilepsy":"Neurological","Dementia":"Neurological",
    "Parkinson’s Disease":"Neurological",

    "Diabetes":"Metabolic","Chronic Kidney Disease":"Metabolic",
    "Thyroid Disorder":"Metabolic","Obesity":"Metabolic",

    "Depression":"Mental Health","Anxiety":"Mental Health",
    "Allergy":"General","Dermatitis":"General","Arthritis":"General","Anemia":"General"
}

df["category"] = df["Disease"].map(DISEASE_TO_CATEGORY)
df.dropna(inplace=True)


nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z, ]", "", text)
    doc = nlp(text)
    return " ".join(
        token.lemma_ for token in doc if not token.is_stop
    )

df["clean_text"] = df["Symptoms"].apply(preprocess) 


le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, stratify=df["label"]
)

class HealthDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels.values

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_ds = HealthDataset(train_enc, y_train)
test_ds  = HealthDataset(test_enc, y_test)  
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)
trainer.train()
trainer.save_model("trained_model")
tokenizer.save_pretrained("trained_model")
import pickle
with open("trained_model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)



