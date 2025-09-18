import schedule
import time
from datetime import datetime

from settings import settings
from database.sql_handler import SQLHandler
from database.vectordb import Vectordb
from agents.news_retrieval import NewsAgent
from agents.sentiment_analysis import SentimentAnalysisAgent
from agents.notification import NotificationAgent


class FinancialSentimentOrchestrator:
    def __init__(self):
        print("🔧 Initializing Financial Sentiment Analysis System...")

        # Initialize components
        self.db_handler = SQLHandler()
        self.vector_store = Vectordb(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index
        )

        # Initialize agents
        self.news_agent = NewsAgent()
        self.sentiment_agent = SentimentAnalysisAgent(self.vector_store)
        self.notification_agent = NotificationAgent()

        print("✅ System initialization completed")

    def run_analysis_cycle(self):
        """Run complete analysis cycle"""
        print(
            f"🔄 Starting analysis cycle at {datetime.now().strftime('%H:%M:%S')}")

        try:
            # Step 1: Fetch news
            articles = self.news_agent.get_indian_financial_news()

            if not articles:
                print("⚠️ No articles retrieved, skipping analysis")
                return []

            print(f"📰 Retrieved {len(articles)} articles")

            # Step 2: Analyze sentiment with RAG
            results = self.sentiment_agent.analyze_articles(articles)

            if not results:
                print("⚠️ No sentiment analysis results")
                return []

            print(f"🧠 Analyzed {len(results)} articles")

            # Step 3: Cache results
            cached_count = 0
            for result in results:
                if self.db_handler.cache_analysis(result):
                    cached_count += 1

            print(f"💾 Cached {cached_count}/{len(results)} results")

            # Step 4: Send instant notifications for important news
            notification_count = 0
            for result in results:
                if (result.sentiment_label.value in ['positive', 'negative'] and
                        result.confidence > settings.high_confidence_threshold):
                    self.notification_agent.send_instant_notification(result)
                    notification_count += 1

            if notification_count > 0:
                print(f"📱 Sent {notification_count} notifications")

            print(
                f"✅ Analysis cycle completed: {len(results)} articles processed")
            return results

        except Exception as e:
            print(f"❌ Error in analysis cycle: {e}")
            return []

    def send_daily_report(self):
        """Generate and send daily email report"""
        print("📊 Generating daily report...")

        try:
            # Get today's cached results
            cached_articles = self.db_handler.get_today_articles()

            if not cached_articles:
                print("⚠️ No articles found for today's report")
                return

            # Convert cached data to SentimentResult objects
            from data_models import NewsArticle, SentimentResult, SentimentLabel

            results = []
            for article_data in cached_articles:
                try:
                    article = NewsArticle(
                        title=article_data['title'],
                        description=article_data['content'],
                        url=article_data['url'],
                        published_at=article_data['published_at'],
                        source=article_data.get('source', 'Unknown')
                    )

                    result = SentimentResult(
                        article=article,
                        sentiment_score=article_data['sentiment'],
                        sentiment_label=SentimentLabel(
                            article_data['sentiment_label']),
                        confidence=article_data.get('confidence', 0.0),
                        context_used=bool(
                            article_data.get('context_used', False))
                    )

                    results.append(result)

                except Exception as e:
                    print(f"⚠️ Error converting cached article: {e}")
                    continue

            # Generate and send report
            report = self.notification_agent.generate_daily_report(results)
            success = self.notification_agent.send_email_report(report)

            if success:
                print("✅ Daily report sent successfully")
            else:
                print("❌ Failed to send daily report")

        except Exception as e:
            print(f"❌ Error generating daily report: {e}")


def run_quick_test():
    """Simple test to verify system works"""
    print("🧪 Running quick system test...")

    try:
        # Test basic initialization
        orchestrator = FinancialSentimentOrchestrator()

        # Test sentiment analysis with dummy article
        from data_models import NewsArticle
        test_article = NewsArticle(
            title="RELIANCE stock surges 10% on strong earnings",
            description="Company reports record quarterly profits",
            url="http://test.com",
            published_at="2025-09-18T10:00:00Z",
            source="test"
        )

        result = orchestrator.sentiment_agent.analyze_single_article(
            test_article)
        print(
            f"✅ Test passed: Detected '{result.sentiment_label.value}' sentiment")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def setup_scheduler():
    """Setup automated scheduling"""
    orchestrator = FinancialSentimentOrchestrator()

    # Run analysis every N hours
    schedule.every(settings.analysis_interval_hours).hours.do(
        orchestrator.run_analysis_cycle
    )

    # Send daily report at specified time
    schedule.every().day.at(settings.daily_report_time).do(
        orchestrator.send_daily_report
    )

    print(f"⏰ Scheduler setup: Analysis every {settings.analysis_interval_hours}h, "
          f"Daily report at {settings.daily_report_time}")

    return orchestrator


def main():
    """Main entry point"""
    print("🚀 FSA using RAG for Indian Stock Market")
    print("=" * 60)

    # Check configuration
    if not settings.news_api_key:
        print("\n❌ Missing NEWS_API_KEY! Check your .env file.")
        return

    try:
        # Run quick test first
        if not run_quick_test():
            print("\n❌ System test failed. Please check your configuration.")
            return

        # Setup and start
        orchestrator = setup_scheduler()

        # Run initial analysis
        print("\n🔄 Running initial analysis...")
        orchestrator.run_analysis_cycle()

        # Start scheduler
        print(f"\n✅ System started successfully!")
        print(
            f"📱 Monitoring Indian financial news every {settings.analysis_interval_hours} hours")
        if settings.email_user:
            print(f"📧 Daily reports will be sent to {settings.email_to}")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 60)

        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        print("\n👋 Financial Sentiment Analysis System stopped gracefully.")
    except Exception as e:
        print(f"\n❌ Fatal system error: {e}")
        print("Check your .env configuration and try again.")


if __name__ == "__main__":
    main()
