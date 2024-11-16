from pydantic import BaseModel
from typing import Optional

class MovieRecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: Optional[int] = 10