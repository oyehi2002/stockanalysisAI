# 🚀 Financial Sentiment Analysis System with RAG

AI-powered multi-agent system for analyzing Indian stock market sentiment using Retrieval Augmented Generation.

## 🎯 Features
- **Real-time News Analysis**: Automated Indian financial news processing
- **RAG-Enhanced AI**: Context-aware sentiment analysis using vector retrieval
- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Automated Reporting**: Daily email reports and instant notifications

## 🏗️ Architecture
- **NewsAgent**: Intelligent news retrieval and filtering
- **SentimentAgent**: FinBERT + RAG for enhanced sentiment analysis  
- **NotificationAgent**: Automated insights and reporting

## 🛠️ Tech Stack
- **AI/ML**: FinBERT, Sentence Transformers, Hugging Face
- **Databases**: SQLite (caching) + Pinecone (vector storage)
- **APIs**: NewsAPI, Gmail SMTP
- **Architecture**: Multi-agent system with RAG pipeline

## 📊 Results
- Processes 20-50 articles per analysis cycle
- Runs automatically every 2 hours
- Generates daily sentiment reports
- Provides instant notifications for high-impact news
