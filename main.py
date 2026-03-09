from fastapi import FastAPI
from model import predict 
from schemas import ModerateRequest, ModerateResponse

app = FastAPI()

@app.get("/")
def check():
    return {"status": "working"}

@app.post("/moderate", response_model=ModerateResponse)
def moderate(request: ModerateRequest):
    response = predict(request.text)

    return ModerateResponse(**response)