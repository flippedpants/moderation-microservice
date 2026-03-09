import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint_path = "./checkpoint-4040"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model.eval()
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def predict(text: str, threshold: float = 0.5):
    inputs = tokenizer(text, return_tensor="pt", padding=True, truncation=True)

    with torch.no_grad():
        output = model(**inputs)

    probs = torch.sigmoid(output.logits).toList()

    scores = {label: round(prob, 3) for label,prob in zip(labels, probs)}
    flagged_labels = [label for label,prob in zip(labels,probs) if prob > threshold]

    