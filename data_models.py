from pydantic import BaseModel, HttpUrl, validator
from datetime import datetime
from typing import Optional, List, Dict, Any
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

    @validator('title', 'description', 'content')
    def clean_text(cls, v):
        if v:
            return v.strip()
        return v


class SentimentResult(BaseModel):
    article: NewsArticle
    sentiment_score: float
    sentiment_label: SentimentLabel
    confidence: float
    context_used: bool
    processed_at: datetime = None

    def __init__(self, **data):
        if 'processed_at' not in data:
            data['processed_at'] = datetime.now()
        super().__init__(**data)

    @validator('sentiment_score')
    def validate_score(cls, v):
        if not -1 <= v <= 1:
            raise ValueError('Sentiment score must be between -1 and 1')
        return v


class AnalysisReport(BaseModel):
    total_articles: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_sentiment: float
    top_positive: List[NewsArticle]
    top_negative: List[NewsArticle]
    generated_at: datetime = None

    def __init__(self, **data):
        if 'generated_at' not in data:
            data['generated_at'] = datetime.now()
        super().__init__(**data)

    @property
    def positive_percentage(self) -> float:
        return (self.positive_count / self.total_articles * 100) if self.total_articles > 0 else 0

    @property
    def negative_percentage(self) -> float:
        return (self.negative_count / self.total_articles * 100) if self.total_articles > 0 else 0

    @property
    def neutral_percentage(self) -> float:
        return (self.neutral_count / self.total_articles * 100) if self.total_articles > 0 else 0
