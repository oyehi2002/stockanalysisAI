import logging
from typing import List
from transformers import pipeline
from data_models import NewsArticle, SentimentResult, SentimentLabel

logger = logging.getLogger(__name__)


class SentimentAnalysisAgent:
    def __init__(self, use_simple_model=False):
        self.use_simple = use_simple_model

        # Initialize lists as instance variables (not global)
        self.positive_articles = []
        self.negative_articles = []
        self.neutral_articles = []

        if not self.use_simple:
            # Load FinBERT for financial sentiment analysis
            self.ai_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"
            )
            print("✅ Loaded FinBERT AI model")

    def analyze_articles(self, articles: List[NewsArticle]) -> List[SentimentResult]:
        """Analyze sentiment for all articles"""
        results = []

        # Clear previous results
        self.positive_articles = []
        self.negative_articles = []
        self.neutral_articles = []

        print(f"🧠 Starting sentiment analysis for {len(articles)} articles...")

        for i, article in enumerate(articles, 1):
            try:
                result = self.analyze_single_article(article)
                results.append(result)
                print(
                    f"✅ {i}/{len(articles)}: {result.sentiment_label.value.upper()} ({result.confidence:.2f})")

            except Exception as e:
                print(f"❌ {i}/{len(articles)}: Failed - {e}")
                continue

        print(
            f"📊 Successfully analyzed {len(results)}/{len(articles)} articles")
        print(f"📈 Positive: {len(self.positive_articles)}")
        print(f"📉 Negative: {len(self.negative_articles)}")
        print(f"😐 Neutral: {len(self.neutral_articles)}")

        return results

    def analyze_single_article(self, article: NewsArticle) -> SentimentResult:
        """Analyze ONE article - choose method based on initialization"""

        # Combine title + description
        text = f"{article.title}. {article.description or ''}".strip()

        if not text:
            raise ValueError("No text to analyze")

        # Truncate if too long
        if len(text) > 400:
            text = text[:400] + "..."

        # Get sentiment analysis results
        sentiment_score, sentiment_label, confidence = self._ai_sentiment(
            text, article)

        return SentimentResult(
            article=article,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            context_used=False
        )

    def _ai_sentiment(self, text: str, article: NewsArticle):
        """FinBERT AI sentiment analysis - better for financial text"""
        raw_result = self.ai_model(text)[0]

        label = raw_result['label'].upper()
        confidence = raw_result['score']

        if label == 'POSITIVE':
            self.positive_articles.append(article.title)
            return confidence, SentimentLabel.POSITIVE, confidence
        elif label == 'NEGATIVE':
            self.negative_articles.append(article.title)
            return -confidence, SentimentLabel.NEGATIVE, confidence
        else:
            self.neutral_articles.append(article.title)
            return 0.0, SentimentLabel.NEUTRAL, confidence

    def get_categorized_articles(self):
        """Get the sorted article lists"""
        return {
            'positive': self.positive_articles,
            'negative': self.negative_articles,
            'neutral': self.neutral_articles
        }
