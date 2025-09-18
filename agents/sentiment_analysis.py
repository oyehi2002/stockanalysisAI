import logging
from typing import List, Tuple
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from data_models import NewsArticle, SentimentResult, SentimentLabel
from database.vectordb import Vectordb
from settings import settings
from helpers import truncate_text

logger = logging.getLogger(__name__)


class SentimentAnalysisAgent:
    def __init__(self, vector_store: Vectordb):
        self.vector_store = vector_store
        self.sentiment_model = None
        self.setup_models()

    def setup_models(self):
        """Initialize sentiment analysis models"""
        try:
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model=settings.sentiment_model,
                tokenizer=settings.sentiment_model
            )
            logger.info(f"Loaded sentiment model: {settings.sentiment_model}")

        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            # Fallback to basic model
            try:
                self.sentiment_model = pipeline("sentiment-analysis")
                logger.info("Using fallback sentiment model")
            except Exception as fallback_error:
                logger.error(
                    f"Failed to load fallback model: {fallback_error}")
                raise

    def analyze_articles(self, articles: List[NewsArticle]) -> List[SentimentResult]:
        """Analyze sentiment for multiple articles with RAG context"""
        results = []

        for article in articles:
            try:
                result = self.analyze_single_article(article)
                if result:
                    results.append(result)

            except Exception as e:
                logger.error(
                    f"Error analyzing article '{article.title[:50]}...': {e}")
                continue

        logger.info(f"Successfully analyzed {len(results)} articles")
        return results

    def analyze_single_article(self, article: NewsArticle) -> SentimentResult:
        """Analyze sentiment for a single article with RAG enhancement"""
        # Prepare text for analysis
        text = f"{article.title} {article.description or ''}".strip()

        if not text:
            raise ValueError("No text content to analyze")

        # Get RAG context
        context = self.vector_store.get_similar_context(
            text,
            similarity_threshold=settings.similarity_threshold
        )

        # Enhanced text with context
        enhanced_text = f"{text}. Context: {context}" if context else text
        enhanced_text = truncate_text(enhanced_text, 512)

        # Analyze sentiment
        sentiment_result = self.sentiment_model(enhanced_text)[0]

        # Convert to standardized format
        sentiment_score, sentiment_label = self._standardize_sentiment(
            sentiment_result)

        # Create result object
        result = SentimentResult(
            article=article,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=sentiment_result.get('score', 0.0),
            context_used=bool(context)
        )

        # Store in vector database
        self.vector_store.store_article(result)

        return result

    def _standardize_sentiment(self, sentiment_result: dict) -> Tuple[float, SentimentLabel]:
        """Convert various sentiment formats to standardized format"""
        label = sentiment_result.get('label', '').upper()
        score = sentiment_result.get('score', 0.0)

        # Map different model outputs to standard format
        if label in ['POSITIVE', 'POS']:
            return score, SentimentLabel.POSITIVE
        elif label in ['NEGATIVE', 'NEG']:
            return -score, SentimentLabel.NEGATIVE  # Negative score for negative sentiment
        else:
            return 0.0, SentimentLabel.NEUTRAL
