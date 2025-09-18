import re
import numpy as np
from typing import List, Dict, Tuple
from data_models import SentimentResult


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)

    return text.strip()


def calculate_sentiment_stats(results: List[SentimentResult]) -> Dict:
    """Calculate sentiment statistics from results"""
    if not results:
        return {
            'total': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'average_score': 0.0,
            'confidence_avg': 0.0
        }

    total = len(results)
    positive = len(
        [r for r in results if r.sentiment_label.value == 'positive'])
    negative = len(
        [r for r in results if r.sentiment_label.value == 'negative'])
    neutral = len([r for r in results if r.sentiment_label.value == 'neutral'])

    scores = [r.sentiment_score for r in results]
    confidences = [r.confidence for r in results]

    return {
        'total': total,
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'average_score': np.mean(scores) if scores else 0.0,
        'confidence_avg': np.mean(confidences) if confidences else 0.0,
        'positive_pct': (positive / total * 100) if total > 0 else 0,
        'negative_pct': (negative / total * 100) if total > 0 else 0,
        'neutral_pct': (neutral / total * 100) if total > 0 else 0
    }


def truncate_text(text: str, max_length: int = 512) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def is_indian_market_relevant(text: str, keywords: List[str]) -> bool:
    """Check if text is relevant to Indian stock market"""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)
