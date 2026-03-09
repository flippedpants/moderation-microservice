import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint_path = "./checkpoint-4040"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

model.eval()
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def predict(text: str, threshold: float = 0.5):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        output = model(**inputs)

    probs = torch.sigmoid(output.logits).squeeze().tolist()

    scores = {label: round(prob, 3) for label,prob in zip(labels, probs)}
    flagged_labels = [label for label,prob in zip(labels,probs) if prob > threshold]

    if flagged_labels:
        primary_label = max(scores, key=scores.get)
        confidence = scores[primary_label]
        flagged = True
    else:
        confidence = round(max(probs), 3)
        flagged = False

    return{
        "text": text,
        "scores": scores,
        "labels": flagged_labels,
        "flagged": flagged,
        "confidence": confidence
    }