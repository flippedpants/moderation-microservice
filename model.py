import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CHECKPOINT_PATH = "./checkpoint-4040"
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer once at module level
print(f"Loading model from {CHECKPOINT_PATH} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
model.to(device)
model.eval()
print("Model loaded successfully.")


def predict(text: str, threshold: float = 0.5) -> dict:
    """
    Run inference on a single text string.

    Returns a dict with:
        - text: the original input text
        - flagged: whether any label exceeded the threshold
        - labels: list of label names above threshold
        - scores: dict of label_name -> confidence score
        - label: the highest-confidence label, or "safe" if none flagged
        - confidence: the highest confidence score among flagged labels (or highest overall if safe)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Multi-label: apply sigmoid to get per-label probabilities
    probabilities = torch.sigmoid(outputs.logits).squeeze().cpu().tolist()

    # Build scores dict
    scores = {label: round(prob, 4) for label, prob in zip(LABELS, probabilities)}

    # Determine which labels are flagged
    flagged_labels = [label for label, prob in zip(LABELS, probabilities) if prob >= threshold]

    # Primary label: highest scoring among flagged, or "safe"
    if flagged_labels:
        primary_label = max(flagged_labels, key=lambda l: scores[l])
        confidence = scores[primary_label]
        flagged = True
    else:
        primary_label = "safe"
        confidence = round(max(probabilities), 4)
        flagged = False

    return {
        "text": text,
        "flagged": flagged,
        "labels": flagged_labels,
        "scores": scores,
        "label": primary_label,
        "confidence": confidence,
    }
