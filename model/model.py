import pandas as pd
from argparse import Namespace
from sklearn.metrics import classification_report
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")

model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def labeling(x):
    label = classifier(x)[0]['label']
    return label
