from fastapi import FastAPI, HTTPException
from schemas import ModerateRequest, ModerateResponse
from model import predict

app = FastAPI(
    title="Content Moderation Microservice",
    description="AI-powered content moderation using a fine-tuned DistilBert model",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/moderate", response_model=ModerateResponse)
def moderate(request: ModerateRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required and cannot be empty")

    result = predict(request.text)
    return ModerateResponse(**result)
