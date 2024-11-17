from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple

class MoviePreferences(BaseModel):
    liked_movies: List[str] = Field(..., min_items=1, description="List of movies the user likes")
    disliked_movies: Optional[List[str]] = Field(default=None, description="Optional list of movies the user dislikes")
    n_recommendations: Optional[int] = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    min_similarity: Optional[float] = Field(default=0.1, ge=0, le=1, description="Minimum similarity threshold")

class MovieRecommendation(BaseModel):
    movie: str
    similarity_score: float