from pydantic import BaseModel
from typing import List

class MovieRecommendationResponse(BaseModel):
    user_id: int
    seen_movies: List[str]
    recommendations: List[str]
    timestamp: str