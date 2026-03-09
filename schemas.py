from pydantic import BaseModel

class ModerateRequest(BaseModel):
    text: str

class ModerateResponse(BaseModel):
    text: str
    scores: dict[str,float]
    labels: list[str]
    flagged: bool
    confidence: float
