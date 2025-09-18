import sqlite3
import logging
from typing import List, Optional
from datetime import datetime
from contextlib import contextmanager
from data_models import NewsArticle, SentimentResult

logger = logging.getLogger(__name__)


class SQLHandler:
    def __init__(self, db_path: str = "financial_news.db"):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT UNIQUE,
                    content TEXT,
                    url TEXT,
                    published_at TEXT,
                    source TEXT,
                    sentiment REAL,
                    sentiment_label TEXT,
                    confidence REAL,
                    context_used BOOLEAN,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_sentiment_processed_at 
                ON news_cache(sentiment_label, processed_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_published_at 
                ON news_cache(published_at)
            ''')

            conn.commit()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def cache_analysis(self, result: SentimentResult) -> bool:
        """Cache sentiment analysis result"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO news_cache 
                    (title, content, url, published_at, source, sentiment, 
                     sentiment_label, confidence, context_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.article.title,
                    result.article.description,
                    result.article.url,
                    result.article.published_at,
                    result.article.source,
                    result.sentiment_score,
                    result.sentiment_label.value,
                    result.confidence,
                    result.context_used
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error caching analysis: {e}")
            return False

    def get_today_articles(self) -> List[dict]:
        """Get today's analyzed articles"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM news_cache 
                    WHERE DATE(processed_at) = DATE('now') 
                    ORDER BY sentiment DESC
                ''')

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching today's articles: {e}")
            return []

    def get_articles_by_sentiment(self, sentiment_label: str, limit: int = 10) -> List[dict]:
        """Get articles by sentiment label"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM news_cache 
                    WHERE sentiment_label = ? 
                    ORDER BY ABS(sentiment) DESC 
                    LIMIT ?
                ''', (sentiment_label, limit))

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching articles by sentiment: {e}")
            return []
