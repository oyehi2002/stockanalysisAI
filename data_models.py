from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from enum import Enum


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class NewsArticle(BaseModel):
    title: str
    description: Optional[str] = None
    content: Optional[str] = None
    url: str
    published_at: str
    source: str


class SentimentResult(BaseModel):
    article: NewsArticle
    sentiment_score: float
    sentiment_label: SentimentLabel
    confidence: float
    context_used: bool = False
    processed_at: datetime = datetime.now()
