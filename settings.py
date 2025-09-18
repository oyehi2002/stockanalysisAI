import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    # API Keys
    news_api_key = os.getenv('NEWS_API_KEY', '')
    pinecone_api_key = os.getenv('PINECONE_API_KEY', '')
    email_user = os.getenv('EMAIL_USER', '')
    email_pass = os.getenv('EMAIL_PASS', '')
    email_to = os.getenv('EMAIL_TO', '')

    # Database
    database_url = os.getenv('DATABASE_URL', 'sqlite:///financial_news.db')

    # Pinecone settings
    pinecone_environment = os.getenv(
        'PINECONE_ENVIRONMENT', 'us-west1-gcp-free')
    pinecone_index = os.getenv('PINECONE_INDEX', 'financial-sentiment')

    # API limits
    max_daily_news = int(os.getenv('MAX_DAILY_NEWS', '100'))

    # Scheduling
    analysis_interval_hours = int(os.getenv('ANALYSIS_INTERVAL_HOURS', '2'))
    daily_report_time = os.getenv('DAILY_REPORT_TIME', '18:00')

    # Model settings
    sentiment_model = os.getenv('SENTIMENT_MODEL', 'ProsusAI/finbert')
    embedding_model = os.getenv(
        'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

    # Indian market keywords
    indian_market_keywords = [
        'BSE', 'NSE', 'SENSEX', 'NIFTY', 'Indian stock market',
        'Mumbai stock exchange', 'National stock exchange',
        'Indian rupee', 'RBI', 'Reserve Bank of India',
        'Tata', 'Reliance', 'Infosys', 'TCS', 'HDFC', 'ICICI',
        'Adani', 'ITC', 'Bharti Airtel', 'Asian Paints'
    ]

    # Thresholds
    high_confidence_threshold = float(
        os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.7'))
    similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))


# Create settings instance
settings = Settings()
