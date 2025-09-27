import requests
import logging
from datetime import datetime, timedelta
from typing import List
import time

from data_models import NewsArticle
from settings import settings

logger = logging.getLogger(__name__)


class NewsAgent:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FinancialSentimentBot/1.0'
        })

    def get_indian_financial_news(self) -> List[NewsArticle]:
        """Fetch financial news with broad search and smart filtering"""
        try:
            url = 'https://newsapi.org/v2/everything'

            broad_queries = [
                'India finance',
                'Indian stock market',
                'SENSEX OR NIFTY',
                'BSE OR NSE',
                'India economy'
            ]

            all_articles = []

            # Trying multiple broad queries to maximize results
            for query in broad_queries:
                try:
                    params = {
                        'q': query,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'from': (datetime.now() - timedelta(days=2)).isoformat(),
                        'apiKey': settings.news_api_key,
                        'pageSize': 20
                    }

                    print(f"🔍 Searching: '{query}'")

                    response = self.session.get(url, params=params, timeout=30)
                    response.raise_for_status()

                    data = response.json()
                    articles = data.get('articles', [])

                    print(f"📊 Found {len(articles)} articles for '{query}'")

                    # Add to collection (will deduplicate later)
                    all_articles.extend(articles)

                    # Avoid rate limiting
                    time.sleep(1)

                except Exception as e:
                    print(f"⚠️ Query '{query}' failed: {e}")
                    continue

            print(f"📈 Total articles collected: {len(all_articles)}")

            # For removing duplicates by URL
            seen_urls = set()
            unique_articles = []
            for article in all_articles:
                url_str = article.get('url', '')
                if url_str not in seen_urls:
                    seen_urls.add(url_str)
                    unique_articles.append(article)

            print(
                f"📰 Unique articles after deduplication: {len(unique_articles)}")

            # To convert to NewsArticle objects and apply SMART FILTERING
            news_articles = []
            for article in unique_articles:
                if self._is_valid_article(article):
                    try:
                        news_article = NewsArticle(
                            title=article.get('title', ''),
                            description=article.get('description', ''),
                            content=article.get('content', ''),
                            url=article.get('url', ''),
                            published_at=article.get('publishedAt', ''),
                            source=article.get('source', {}).get('name', '')
                        )

                        # SMART FILTERING - Multiple relevance checks
                        if self._is_financially_relevant(news_article):
                            news_articles.append(news_article)

                    except Exception as e:
                        continue

            print(f"✅ Final relevant articles: {len(news_articles)}")
            return news_articles

        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return []

    def _is_valid_article(self, article: dict) -> bool:
        """Validate article has required fields"""
        required_fields = ['title', 'url']
        return all(article.get(field) for field in required_fields)

    def _is_financially_relevant(self, article: NewsArticle) -> bool:
        """Enhanced relevance checking with multiple criteria"""
        text = f"{article.title} {article.description or ''}".lower()

        # Financial keywords (broader than just Indian market)
        financial_indicators = [
            'stock', 'market', 'shares', 'trading', 'finance', 'investment',
            'economy', 'rupee', 'bank', 'bse', 'nse', 'sensex', 'nifty',
            'earnings', 'profit', 'revenue', 'ipo', 'dividend'
        ]

        # Indian context indicators
        indian_indicators = [
            'india', 'indian', 'mumbai', 'delhi', 'bangalore', 'rbi',
            'reserve bank', 'tata', 'reliance', 'infosys', 'hdfc',
            'adani', 'icici', 'wipro', 'bharti'
        ]

        # Check if article has financial relevance
        has_financial = any(
            indicator in text for indicator in financial_indicators)

        # Check if article has Indian context
        has_indian_context = any(
            indicator in text for indicator in indian_indicators)

        # Accept if it has both financial relevance AND Indian context
        # OR if it has strong Indian market keywords
        strong_indian_market = any(keyword.lower() in text for keyword in [
            'sensex', 'nifty', 'bse', 'nse', 'indian stock market'
        ])

        return (has_financial and has_indian_context) or strong_indian_market
