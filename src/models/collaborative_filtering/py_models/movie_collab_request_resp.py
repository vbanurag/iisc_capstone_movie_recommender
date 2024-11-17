from pydantic import BaseModel
from typing import List, Dict
from .movie_recommendation import MovieRecommendation

class RecommendationResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    processing_time: float

class ModelStatus(BaseModel):
    status: str
    message: str