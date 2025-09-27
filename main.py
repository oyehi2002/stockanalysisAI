import schedule
import time
from datetime import datetime
from settings import settings
from agents import NewsAgent
from agents import SentimentAnalysisAgent
from agents import NotificationAgent


class FinancialSentimentOrchestrator:
    def __init__(self):
        print("🔧 Initializing Financial Sentiment Analysis System...")

        # Initialize agents
        self.news_agent = NewsAgent()
        self.sentiment_agent = SentimentAnalysisAgent()
        self.notification_agent = NotificationAgent()

        print("✅ System ready!")

    def run_analysis_cycle(self):
        """Run complete analysis cycle - fetch, analyze, notify"""
        print(f"🔄 Starting analysis at {datetime.now().strftime('%H:%M:%S')}")

        try:
            # Step 1: Get news articles
            articles = self.news_agent.get_indian_financial_news()

            if not articles:
                print("⚠️ No articles found, skipping...")
                return

            print(f"📰 Found {len(articles)} articles")

            # Step 2: Analyze sentiment (populates categorized lists)
            results = self.sentiment_agent.analyze_articles(articles)

            if not results:
                print("⚠️ No analysis results")
                return

            # Step 3: Send all notifications (desktop + email)
            self.notification_agent.send_all_notifications(
                self.sentiment_agent, articles)

            print("✅ Analysis cycle completed!")

        except Exception as e:
            print(f"❌ Error in analysis: {e}")

    def run_once(self):
        """Run analysis once for testing"""
        print("🧪 Running single analysis...")
        self.run_analysis_cycle()


def setup_scheduler():
    """Setup automated scheduling"""
    orchestrator = FinancialSentimentOrchestrator()

    # Run analysis every N hours (from settings)
    schedule.every(settings.analysis_interval_hours).hours.do(
        orchestrator.run_analysis_cycle
    )

    print(
        f"⏰ Scheduled: Analysis every {settings.analysis_interval_hours} hours")
    return orchestrator


def main():
    """Main entry point"""
    print("🚀 Financial Sentiment Analysis for Indian Stock Market")
    print("=" * 60)

    # Check API key
    if not settings.news_api_key:
        print("❌ Missing NEWS_API_KEY in .env file!")
        return

    try:
        # Quick test
        print("🧪 Testing system...")
        orchestrator = FinancialSentimentOrchestrator()

        # Option 1: Run once and exit
        print("\n[1] Run once and exit")
        print("[2] Run continuously in background")
        choice = input("Choose option (1 or 2): ").strip()

        if choice == "1":
            orchestrator.run_once()
            print("👋 Single run completed!")

        elif choice == "2":
            # Setup scheduler
            orchestrator = setup_scheduler()

            # Run initial analysis
            print("\n🔄 Running initial analysis...")
            orchestrator.run_analysis_cycle()

            # Start continuous monitoring
            print(f"\n✅ System started!")
            print(
                f"📱 Monitoring every {settings.analysis_interval_hours} hours")
            print("🛑 Press Ctrl+C to stop")
            print("=" * 60)

            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        else:
            print("Invalid choice!")

    except KeyboardInterrupt:
        print("\n👋 System stopped!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
