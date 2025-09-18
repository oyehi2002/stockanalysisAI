import logging
from typing import List, Dict, Optional
import numpy as np

try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

from sentence_transformers import SentenceTransformer
from data_models import NewsArticle, SentimentResult

logger = logging.getLogger(__name__)


class Vectordb:
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = None
        self.index = None
        self.embedding_model = None
        self.setup_pinecone()

    def setup_pinecone(self):
        """Initialize Pinecone vector database"""
        if not HAS_PINECONE:
            print("⚠️ Pinecone not installed. Vector store functionality disabled.")
            print("   RAG features will be limited but system will still work!")
            return

        if not self.api_key:
            print("⚠️ No Pinecone API key found. Vector store functionality disabled.")
            print("   RAG features will be limited but system will still work!")
            return

        try:
            # New Pinecone API
            self.pc = Pinecone(api_key=self.api_key)

            # Create index if it doesn't exist
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                print(f"🔧 Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Free tier region
                    )
                )
                print(f"✅ Created Pinecone index: {self.index_name}")

            self.index = self.pc.Index(self.index_name)
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2'
            )
            print("✅ Pinecone setup completed successfully")

        except Exception as e:
            print(f"⚠️ Error setting up Pinecone: {e}")
            print("   Vector store functionality disabled but system will still work!")
            self.pc = None
            self.index = None

    def store_article(self, result: SentimentResult) -> bool:
        """Store article embedding and metadata"""
        if not self.index or not self.embedding_model:
            return False

        try:
            text = f"{result.article.title} {result.article.description or ''}"
            embedding = self.embedding_model.encode([text])[0].tolist()

            # Create unique ID
            article_id = f"article_{hash(result.article.url)}"

            # Store in Pinecone
            self.index.upsert([
                (article_id, embedding, {
                    'title': result.article.title,
                    'url': result.article.url,
                    'sentiment_score': result.sentiment_score,
                    'sentiment_label': result.sentiment_label.value,
                    'published_at': result.article.published_at,
                    'source': result.article.source,
                    'confidence': result.confidence
                })
            ])

            return True

        except Exception as e:
            logger.error(f"Error storing article in vector database: {e}")
            return False

    def get_similar_context(self, text: str, top_k: int = 3,
                            similarity_threshold: float = 0.7) -> str:
        """Get relevant context from vector database"""
        if not self.index or not self.embedding_model:
            return ""

        try:
            # Get embedding for the query
            query_embedding = self.embedding_model.encode([text])[0].tolist()

            # Query similar articles
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Extract relevant context
            contexts = []
            for match in results.get('matches', []):
                if match.get('score', 0) > similarity_threshold:
                    metadata = match.get('metadata', {})
                    sentiment = metadata.get('sentiment_label', 'neutral')
                    contexts.append(f"Similar news was {sentiment}")

            return " ".join(contexts) if contexts else ""

        except Exception as e:
            logger.error(f"Error getting similar context: {e}")
            return ""

    def get_index_stats(self) -> Dict:
        """Get vector database statistics"""
        if not self.index:
            return {}

        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
