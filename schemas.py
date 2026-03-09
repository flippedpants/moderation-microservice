from pydantic import BaseModel


class ModerateRequest(BaseModel):
    text: str


class ModerateResponse(BaseModel):
    text: str
    flagged: bool
    labels: list[str]
    scores: dict[str, float]
    label: str
    confidence: float
