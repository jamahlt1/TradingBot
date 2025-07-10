import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
import requests
import aiohttp
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import yfinance as yf
import tweepy
import praw
from bs4 import BeautifulSoup
import feedparser
import openai

logger = logging.getLogger(__name__)

class SentimentSource(Enum):
    """Sentiment Data Sources"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    REDDIT = "reddit"
    TWITTER = "twitter"
    FORUMS = "forums"
    BLOGS = "blogs"
    ANALYST_REPORTS = "analyst_reports"
    EARNINGS_CALLS = "earnings_calls"
    SEC_FILINGS = "sec_filings"
    PRESS_RELEASES = "press_releases"

class SentimentType(Enum):
    """Sentiment Types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class SentimentData:
    """Sentiment Data Point"""
    id: str
    source: SentimentSource
    content: str
    title: Optional[str]
    author: Optional[str]
    timestamp: datetime
    symbol: Optional[str]
    sentiment_score: float
    sentiment_type: SentimentType
    confidence: float
    keywords: List[str]
    entities: List[str]
    topics: List[str]
    language: str
    url: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class SentimentAnalysis:
    """Comprehensive Sentiment Analysis Result"""
    symbol: str
    timestamp: datetime
    overall_sentiment: float
    sentiment_type: SentimentType
    confidence: float
    
    # Source-specific sentiment
    news_sentiment: float
    social_sentiment: float
    reddit_sentiment: float
    twitter_sentiment: float
    
    # Temporal analysis
    sentiment_trend: float
    sentiment_momentum: float
    sentiment_volatility: float
    
    # Volume analysis
    sentiment_volume: int
    positive_volume: int
    negative_volume: int
    neutral_volume: int
    
    # Topic analysis
    top_topics: List[str]
    topic_sentiment: Dict[str, float]
    
    # Entity analysis
    top_entities: List[str]
    entity_sentiment: Dict[str, float]
    
    # Market correlation
    price_correlation: float
    volume_correlation: float
    
    # Predictive signals
    bullish_signals: List[str]
    bearish_signals: List[str]
    
    # Raw data
    sentiment_data: List[SentimentData]
    
    # Metadata
    analysis_duration: float
    data_sources: List[str]
    model_confidence: float

class SentimentAnalyzer:
    """
    Comprehensive Sentiment Analysis System
    - Multi-source data collection
    - Advanced NLP processing
    - Real-time sentiment analysis
    - Market correlation analysis
    - Predictive sentiment signals
    """
    
    def __init__(self,
                 api_keys: Dict[str, str],
                 nlp_model: str = "en_core_web_sm",
                 sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                 enable_llm: bool = True,
                 enable_news_api: bool = True,
                 enable_social_media: bool = True,
                 enable_reddit: bool = True,
                 enable_twitter: bool = True):
        
        self.api_keys = api_keys
        self.nlp_model = nlp_model
        self.sentiment_model = sentiment_model
        self.enable_llm = enable_llm
        self.enable_news_api = enable_news_api
        self.enable_social_media = enable_social_media
        self.enable_reddit = enable_reddit
        self.enable_twitter = enable_twitter
        
        # Initialize NLP components
        self.nlp = spacy.load(nlp_model)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize transformer models
        self.tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(sentiment_model)
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model)
        
        # Initialize LLM if enabled
        if self.enable_llm and 'openai' in api_keys:
            self.openai_client = openai.OpenAI(api_key=api_keys['openai'])
        else:
            self.openai_client = None
        
        # Initialize data sources
        self._initialize_data_sources()
        
        # Sentiment cache
        self.sentiment_cache = {}
        self.sentiment_history = {}
        
        # Keywords and entities
        self.financial_keywords = self._load_financial_keywords()
        self.company_entities = self._load_company_entities()
        
        logger.info("Sentiment Analyzer initialized")
    
    def _initialize_data_sources(self):
        """Initialize data source connections"""
        try:
            # News API
            if self.enable_news_api and 'news_api' in self.api_keys:
                self.news_api_key = self.api_keys['news_api']
            
            # Twitter API
            if self.enable_twitter and 'twitter' in self.api_keys:
                twitter_config = self.api_keys['twitter']
                self.twitter_auth = tweepy.OAuthHandler(
                    twitter_config['consumer_key'],
                    twitter_config['consumer_secret']
                )
                self.twitter_auth.set_access_token(
                    twitter_config['access_token'],
                    twitter_config['access_token_secret']
                )
                self.twitter_api = tweepy.API(self.twitter_auth)
            
            # Reddit API
            if self.enable_reddit and 'reddit' in self.api_keys:
                reddit_config = self.api_keys['reddit']
                self.reddit = praw.Reddit(
                    client_id=reddit_config['client_id'],
                    client_secret=reddit_config['client_secret'],
                    user_agent=reddit_config['user_agent']
                )
            
            logger.info("Data sources initialized")
            
        except Exception as e:
            logger.error(f"Error initializing data sources: {e}")
    
    def _load_financial_keywords(self) -> List[str]:
        """Load financial keywords for analysis"""
        try:
            keywords = [
                # Market terms
                'bull', 'bear', 'rally', 'crash', 'correction', 'bubble', 'recession',
                'inflation', 'deflation', 'interest rates', 'fed', 'central bank',
                
                # Trading terms
                'buy', 'sell', 'hold', 'long', 'short', 'position', 'portfolio',
                'dividend', 'earnings', 'revenue', 'profit', 'loss', 'margin',
                
                # Company terms
                'ceo', 'cfo', 'board', 'shareholders', 'stakeholders', 'merger',
                'acquisition', 'ipo', 'secondary offering', 'buyback', 'split',
                
                # Technical terms
                'support', 'resistance', 'breakout', 'breakdown', 'trend', 'momentum',
                'volume', 'volatility', 'beta', 'alpha', 'sharpe ratio', 'rsi',
                
                # Economic terms
                'gdp', 'unemployment', 'inflation', 'deflation', 'monetary policy',
                'fiscal policy', 'trade deficit', 'current account', 'debt', 'surplus'
            ]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error loading financial keywords: {e}")
            return []
    
    def _load_company_entities(self) -> Dict[str, List[str]]:
        """Load company entities and aliases"""
        try:
            entities = {
                'AAPL': ['Apple', 'Apple Inc', 'iPhone', 'iPad', 'Mac', 'Tim Cook'],
                'GOOGL': ['Google', 'Alphabet', 'YouTube', 'Android', 'Sundar Pichai'],
                'MSFT': ['Microsoft', 'Windows', 'Office', 'Azure', 'Satya Nadella'],
                'AMZN': ['Amazon', 'AWS', 'Jeff Bezos', 'Andy Jassy', 'Prime'],
                'TSLA': ['Tesla', 'Elon Musk', 'Model S', 'Model 3', 'Model X', 'Model Y'],
                'META': ['Facebook', 'Meta', 'Instagram', 'WhatsApp', 'Mark Zuckerberg'],
                'NVDA': ['NVIDIA', 'Jensen Huang', 'GPU', 'AI', 'gaming'],
                'NFLX': ['Netflix', 'streaming', 'Reed Hastings', 'Ted Sarandos'],
                'CRM': ['Salesforce', 'Marc Benioff', 'cloud', 'CRM'],
                'AMD': ['Advanced Micro Devices', 'Lisa Su', 'CPU', 'GPU']
            }
            
            return entities
            
        except Exception as e:
            logger.error(f"Error loading company entities: {e}")
            return {}
    
    async def analyze_sentiment(self, symbol: str, timeframe: str = "1d") -> SentimentAnalysis:
        """Perform comprehensive sentiment analysis for a symbol"""
        try:
            start_time = time.time()
            
            # Collect sentiment data from multiple sources
            sentiment_data = await self._collect_sentiment_data(symbol, timeframe)
            
            # Process and analyze sentiment
            analysis = await self._process_sentiment_data(symbol, sentiment_data)
            
            # Calculate market correlations
            analysis = await self._calculate_market_correlations(symbol, analysis)
            
            # Generate predictive signals
            analysis = await self._generate_predictive_signals(analysis)
            
            # Update cache and history
            self._update_sentiment_cache(symbol, analysis)
            
            analysis.analysis_duration = time.time() - start_time
            
            logger.info(f"Sentiment analysis completed for {symbol}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            raise
    
    async def _collect_sentiment_data(self, symbol: str, timeframe: str) -> List[SentimentData]:
        """Collect sentiment data from multiple sources"""
        try:
            sentiment_data = []
            
            # Collect from news sources
            if self.enable_news_api:
                news_data = await self._collect_news_data(symbol)
                sentiment_data.extend(news_data)
            
            # Collect from social media
            if self.enable_social_media:
                social_data = await self._collect_social_media_data(symbol)
                sentiment_data.extend(social_data)
            
            # Collect from Reddit
            if self.enable_reddit:
                reddit_data = await self._collect_reddit_data(symbol)
                sentiment_data.extend(reddit_data)
            
            # Collect from Twitter
            if self.enable_twitter:
                twitter_data = await self._collect_twitter_data(symbol)
                sentiment_data.extend(twitter_data)
            
            # Collect from RSS feeds
            rss_data = await self._collect_rss_data(symbol)
            sentiment_data.extend(rss_data)
            
            # Collect from financial websites
            financial_data = await self._collect_financial_data(symbol)
            sentiment_data.extend(financial_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting sentiment data: {e}")
            return []
    
    async def _collect_news_data(self, symbol: str) -> List[SentimentData]:
        """Collect news sentiment data"""
        try:
            news_data = []
            
            # Get company name and keywords
            company_keywords = self._get_company_keywords(symbol)
            
            async with aiohttp.ClientSession() as session:
                for keyword in company_keywords:
                    # News API
                    if hasattr(self, 'news_api_key'):
                        url = f"https://newsapi.org/v2/everything"
                        params = {
                            'q': keyword,
                            'apiKey': self.news_api_key,
                            'language': 'en',
                            'sortBy': 'publishedAt',
                            'pageSize': 50
                        }
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                for article in data.get('articles', []):
                                    sentiment_data = await self._process_article(article, symbol, SentimentSource.NEWS)
                                    if sentiment_data:
                                        news_data.append(sentiment_data)
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return []
    
    async def _collect_social_media_data(self, symbol: str) -> List[SentimentData]:
        """Collect social media sentiment data"""
        try:
            social_data = []
            
            # This would integrate with various social media APIs
            # For now, return empty list
            return social_data
            
        except Exception as e:
            logger.error(f"Error collecting social media data: {e}")
            return []
    
    async def _collect_reddit_data(self, symbol: str) -> List[SentimentData]:
        """Collect Reddit sentiment data"""
        try:
            reddit_data = []
            
            if hasattr(self, 'reddit'):
                # Get company keywords
                company_keywords = self._get_company_keywords(symbol)
                
                # Search in relevant subreddits
                subreddits = ['investing', 'stocks', 'wallstreetbets', 'StockMarket']
                
                for subreddit_name in subreddits:
                    try:
                        subreddit = self.reddit.subreddit(subreddit_name)
                        
                        for keyword in company_keywords:
                            # Search for posts
                            posts = subreddit.search(keyword, limit=20, sort='hot')
                            
                            for post in posts:
                                # Process post title and content
                                content = f"{post.title} {post.selftext}"
                                
                                sentiment_data = await self._process_text_content(
                                    content, symbol, SentimentSource.REDDIT,
                                    title=post.title,
                                    author=post.author.name if post.author else None,
                                    timestamp=datetime.fromtimestamp(post.created_utc),
                                    url=f"https://reddit.com{post.permalink}"
                                )
                                
                                if sentiment_data:
                                    reddit_data.append(sentiment_data)
                                
                                # Process comments
                                post.comments.replace_more(limit=0)
                                for comment in post.comments.list()[:10]:
                                    comment_sentiment = await self._process_text_content(
                                        comment.body, symbol, SentimentSource.REDDIT,
                                        author=comment.author.name if comment.author else None,
                                        timestamp=datetime.fromtimestamp(comment.created_utc)
                                    )
                                    
                                    if comment_sentiment:
                                        reddit_data.append(comment_sentiment)
                    
                    except Exception as e:
                        logger.error(f"Error processing subreddit {subreddit_name}: {e}")
                        continue
            
            return reddit_data
            
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
            return []
    
    async def _collect_twitter_data(self, symbol: str) -> List[SentimentData]:
        """Collect Twitter sentiment data"""
        try:
            twitter_data = []
            
            if hasattr(self, 'twitter_api'):
                # Get company keywords
                company_keywords = self._get_company_keywords(symbol)
                
                for keyword in company_keywords:
                    try:
                        # Search tweets
                        tweets = self.twitter_api.search_tweets(
                            q=keyword,
                            lang='en',
                            count=100,
                            tweet_mode='extended'
                        )
                        
                        for tweet in tweets:
                            sentiment_data = await self._process_text_content(
                                tweet.full_text, symbol, SentimentSource.TWITTER,
                                author=tweet.user.screen_name,
                                timestamp=tweet.created_at,
                                url=f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
                            )
                            
                            if sentiment_data:
                                twitter_data.append(sentiment_data)
                    
                    except Exception as e:
                        logger.error(f"Error processing Twitter keyword {keyword}: {e}")
                        continue
            
            return twitter_data
            
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {e}")
            return []
    
    async def _collect_rss_data(self, symbol: str) -> List[SentimentData]:
        """Collect RSS feed data"""
        try:
            rss_data = []
            
            # Financial RSS feeds
            rss_feeds = [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://www.marketwatch.com/rss/topstories',
                'https://www.cnbc.com/id/100003114/device/rss/rss.html'
            ]
            
            async with aiohttp.ClientSession() as session:
                for feed_url in rss_feeds:
                    try:
                        async with session.get(feed_url) as response:
                            if response.status == 200:
                                content = await response.text()
                                feed = feedparser.parse(content)
                                
                                for entry in feed.entries[:20]:
                                    # Check if entry is relevant to symbol
                                    if self._is_relevant_to_symbol(entry.title + " " + entry.summary, symbol):
                                        sentiment_data = await self._process_article(
                                            {
                                                'title': entry.title,
                                                'description': entry.summary,
                                                'publishedAt': entry.published,
                                                'url': entry.link
                                            },
                                            symbol,
                                            SentimentSource.NEWS
                                        )
                                        
                                        if sentiment_data:
                                            rss_data.append(sentiment_data)
                    
                    except Exception as e:
                        logger.error(f"Error processing RSS feed {feed_url}: {e}")
                        continue
            
            return rss_data
            
        except Exception as e:
            logger.error(f"Error collecting RSS data: {e}")
            return []
    
    async def _collect_financial_data(self, symbol: str) -> List[SentimentData]:
        """Collect financial website data"""
        try:
            financial_data = []
            
            # Yahoo Finance
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Process company description
                if 'longBusinessSummary' in info:
                    sentiment_data = await self._process_text_content(
                        info['longBusinessSummary'], symbol, SentimentSource.NEWS,
                        title=f"{symbol} Company Description",
                        timestamp=datetime.now()
                    )
                    
                    if sentiment_data:
                        financial_data.append(sentiment_data)
            
            except Exception as e:
                logger.error(f"Error collecting Yahoo Finance data: {e}")
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error collecting financial data: {e}")
            return []
    
    async def _process_article(self, article: Dict[str, Any], symbol: str, source: SentimentSource) -> Optional[SentimentData]:
        """Process news article"""
        try:
            content = f"{article.get('title', '')} {article.get('description', '')}"
            
            return await self._process_text_content(
                content, symbol, source,
                title=article.get('title'),
                timestamp=datetime.fromisoformat(article.get('publishedAt', datetime.now().isoformat())),
                url=article.get('url')
            )
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
    
    async def _process_text_content(self, content: str, symbol: str, source: SentimentSource,
                                  title: Optional[str] = None, author: Optional[str] = None,
                                  timestamp: Optional[datetime] = None, url: Optional[str] = None) -> Optional[SentimentData]:
        """Process text content and extract sentiment"""
        try:
            # Clean and preprocess text
            cleaned_content = self._preprocess_text(content)
            
            if not cleaned_content or len(cleaned_content) < 10:
                return None
            
            # Analyze sentiment using multiple methods
            sentiment_scores = await self._analyze_sentiment_multiple_methods(cleaned_content)
            
            # Extract keywords and entities
            keywords = self._extract_keywords(cleaned_content)
            entities = self._extract_entities(cleaned_content)
            topics = self._extract_topics(cleaned_content)
            
            # Determine overall sentiment
            overall_sentiment = np.mean([
                sentiment_scores['vader'],
                sentiment_scores['textblob'],
                sentiment_scores['transformer']
            ])
            
            sentiment_type = self._classify_sentiment(overall_sentiment)
            confidence = self._calculate_confidence(sentiment_scores)
            
            # Create sentiment data
            sentiment_data = SentimentData(
                id=f"{source.value}_{timestamp.timestamp() if timestamp else time.time()}",
                source=source,
                content=cleaned_content,
                title=title,
                author=author,
                timestamp=timestamp or datetime.now(),
                symbol=symbol,
                sentiment_score=overall_sentiment,
                sentiment_type=sentiment_type,
                confidence=confidence,
                keywords=keywords,
                entities=entities,
                topics=topics,
                language='en',
                url=url,
                metadata={
                    'vader_score': sentiment_scores['vader'],
                    'textblob_score': sentiment_scores['textblob'],
                    'transformer_score': sentiment_scores['transformer'],
                    'llm_score': sentiment_scores.get('llm', 0)
                }
            )
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error processing text content: {e}")
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        try:
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove special characters but keep important ones
            text = re.sub(r'[^\w\s\.\,\!\?\-\$\%]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Convert to lowercase
            text = text.lower()
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    async def _analyze_sentiment_multiple_methods(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using multiple methods"""
        try:
            scores = {}
            
            # VADER Sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            scores['vader'] = vader_scores['compound']
            
            # TextBlob Sentiment
            blob = TextBlob(text)
            scores['textblob'] = blob.sentiment.polarity
            
            # Transformer Model
            try:
                result = self.sentiment_pipeline(text)
                if result[0]['label'] == 'POSITIVE':
                    scores['transformer'] = result[0]['score']
                elif result[0]['label'] == 'NEGATIVE':
                    scores['transformer'] = -result[0]['score']
                else:
                    scores['transformer'] = 0
            except Exception as e:
                logger.error(f"Error with transformer sentiment: {e}")
                scores['transformer'] = 0
            
            # LLM Analysis (if enabled)
            if self.openai_client:
                try:
                    llm_score = await self._analyze_sentiment_llm(text)
                    scores['llm'] = llm_score
                except Exception as e:
                    logger.error(f"Error with LLM sentiment: {e}")
                    scores['llm'] = 0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'vader': 0, 'textblob': 0, 'transformer': 0}
    
    async def _analyze_sentiment_llm(self, text: str) -> float:
        """Analyze sentiment using LLM"""
        try:
            prompt = f"""
            Analyze the sentiment of the following financial text. 
            Return a score between -1 (very negative) and 1 (very positive).
            Focus on financial and market implications.
            
            Text: {text}
            
            Score: """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
            )
            
            score_text = response.choices[0].message.content.strip()
            
            try:
                score = float(score_text)
                return max(-1, min(1, score))  # Clamp between -1 and 1
            except ValueError:
                return 0
                
        except Exception as e:
            logger.error(f"Error with LLM sentiment analysis: {e}")
            return 0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        try:
            # Tokenize and lemmatize
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            
            # Find financial keywords
            keywords = []
            for token in tokens:
                if token in self.financial_keywords:
                    keywords.append(token)
            
            return keywords[:10]  # Return top 10 keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'GPE', 'MONEY']:
                    entities.append(ent.text)
            
            return entities[:10]  # Return top 10 entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using LDA"""
        try:
            # Simple topic extraction based on financial terms
            topics = []
            
            # Check for common financial topics
            topic_keywords = {
                'earnings': ['earnings', 'revenue', 'profit', 'loss', 'quarterly'],
                'dividend': ['dividend', 'payout', 'yield', 'distribution'],
                'acquisition': ['merger', 'acquisition', 'buyout', 'takeover'],
                'ipo': ['ipo', 'initial public offering', 'listing'],
                'analyst': ['analyst', 'rating', 'target', 'upgrade', 'downgrade'],
                'insider': ['insider', 'executive', 'ceo', 'cfo', 'director'],
                'regulatory': ['sec', 'regulation', 'compliance', 'investigation'],
                'market': ['market', 'trading', 'volume', 'price', 'stock']
            }
            
            text_lower = text.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def _classify_sentiment(self, score: float) -> SentimentType:
        """Classify sentiment based on score"""
        if score > 0.1:
            return SentimentType.POSITIVE
        elif score < -0.1:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate confidence in sentiment analysis"""
        try:
            # Calculate standard deviation of scores
            values = list(scores.values())
            if len(values) > 1:
                std_dev = np.std(values)
                # Higher confidence for lower standard deviation
                confidence = max(0.1, 1 - std_dev)
            else:
                confidence = 0.5
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _get_company_keywords(self, symbol: str) -> List[str]:
        """Get keywords for company search"""
        try:
            keywords = [symbol]
            
            # Add company entities if available
            if symbol in self.company_entities:
                keywords.extend(self.company_entities[symbol])
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error getting company keywords: {e}")
            return [symbol]
    
    def _is_relevant_to_symbol(self, text: str, symbol: str) -> bool:
        """Check if text is relevant to symbol"""
        try:
            text_lower = text.lower()
            symbol_lower = symbol.lower()
            
            # Check for symbol mention
            if symbol_lower in text_lower:
                return True
            
            # Check for company entities
            if symbol in self.company_entities:
                for entity in self.company_entities[symbol]:
                    if entity.lower() in text_lower:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return False
    
    async def _process_sentiment_data(self, symbol: str, sentiment_data: List[SentimentData]) -> SentimentAnalysis:
        """Process collected sentiment data"""
        try:
            if not sentiment_data:
                return self._create_empty_analysis(symbol)
            
            # Calculate overall sentiment
            overall_sentiment = np.mean([data.sentiment_score for data in sentiment_data])
            sentiment_type = self._classify_sentiment(overall_sentiment)
            confidence = np.mean([data.confidence for data in sentiment_data])
            
            # Calculate source-specific sentiment
            source_sentiment = self._calculate_source_sentiment(sentiment_data)
            
            # Calculate temporal analysis
            temporal_analysis = self._calculate_temporal_analysis(sentiment_data)
            
            # Calculate volume analysis
            volume_analysis = self._calculate_volume_analysis(sentiment_data)
            
            # Calculate topic analysis
            topic_analysis = self._calculate_topic_analysis(sentiment_data)
            
            # Calculate entity analysis
            entity_analysis = self._calculate_entity_analysis(sentiment_data)
            
            # Create analysis result
            analysis = SentimentAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_sentiment=overall_sentiment,
                sentiment_type=sentiment_type,
                confidence=confidence,
                news_sentiment=source_sentiment.get('news', 0),
                social_sentiment=source_sentiment.get('social_media', 0),
                reddit_sentiment=source_sentiment.get('reddit', 0),
                twitter_sentiment=source_sentiment.get('twitter', 0),
                sentiment_trend=temporal_analysis['trend'],
                sentiment_momentum=temporal_analysis['momentum'],
                sentiment_volatility=temporal_analysis['volatility'],
                sentiment_volume=volume_analysis['total'],
                positive_volume=volume_analysis['positive'],
                negative_volume=volume_analysis['negative'],
                neutral_volume=volume_analysis['neutral'],
                top_topics=topic_analysis['top_topics'],
                topic_sentiment=topic_analysis['topic_sentiment'],
                top_entities=entity_analysis['top_entities'],
                entity_sentiment=entity_analysis['entity_sentiment'],
                price_correlation=0,  # Will be calculated later
                volume_correlation=0,  # Will be calculated later
                bullish_signals=[],  # Will be generated later
                bearish_signals=[],  # Will be generated later
                sentiment_data=sentiment_data,
                analysis_duration=0,
                data_sources=list(set([data.source.value for data in sentiment_data])),
                model_confidence=confidence
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing sentiment data: {e}")
            return self._create_empty_analysis(symbol)
    
    def _calculate_source_sentiment(self, sentiment_data: List[SentimentData]) -> Dict[str, float]:
        """Calculate sentiment by source"""
        try:
            source_sentiment = {}
            
            for source in SentimentSource:
                source_data = [data for data in sentiment_data if data.source == source]
                if source_data:
                    source_sentiment[source.value] = np.mean([data.sentiment_score for data in source_data])
                else:
                    source_sentiment[source.value] = 0
            
            return source_sentiment
            
        except Exception as e:
            logger.error(f"Error calculating source sentiment: {e}")
            return {}
    
    def _calculate_temporal_analysis(self, sentiment_data: List[SentimentData]) -> Dict[str, float]:
        """Calculate temporal sentiment analysis"""
        try:
            # Sort by timestamp
            sorted_data = sorted(sentiment_data, key=lambda x: x.timestamp)
            
            if len(sorted_data) < 2:
                return {'trend': 0, 'momentum': 0, 'volatility': 0}
            
            # Calculate trend (linear regression slope)
            timestamps = [(data.timestamp - sorted_data[0].timestamp).total_seconds() for data in sorted_data]
            scores = [data.sentiment_score for data in sorted_data]
            
            if len(set(timestamps)) > 1:
                trend = np.polyfit(timestamps, scores, 1)[0]
            else:
                trend = 0
            
            # Calculate momentum (recent vs historical)
            if len(sorted_data) >= 10:
                recent = sorted_data[-10:]
                historical = sorted_data[:-10]
                
                recent_avg = np.mean([data.sentiment_score for data in recent])
                historical_avg = np.mean([data.sentiment_score for data in historical])
                momentum = recent_avg - historical_avg
            else:
                momentum = 0
            
            # Calculate volatility
            volatility = np.std(scores)
            
            return {
                'trend': trend,
                'momentum': momentum,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating temporal analysis: {e}")
            return {'trend': 0, 'momentum': 0, 'volatility': 0}
    
    def _calculate_volume_analysis(self, sentiment_data: List[SentimentData]) -> Dict[str, int]:
        """Calculate sentiment volume analysis"""
        try:
            total = len(sentiment_data)
            positive = len([data for data in sentiment_data if data.sentiment_type == SentimentType.POSITIVE])
            negative = len([data for data in sentiment_data if data.sentiment_type == SentimentType.NEGATIVE])
            neutral = len([data for data in sentiment_data if data.sentiment_type == SentimentType.NEUTRAL])
            
            return {
                'total': total,
                'positive': positive,
                'negative': negative,
                'neutral': neutral
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume analysis: {e}")
            return {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
    
    def _calculate_topic_analysis(self, sentiment_data: List[SentimentData]) -> Dict[str, Any]:
        """Calculate topic-based sentiment analysis"""
        try:
            topic_sentiment = {}
            topic_counts = {}
            
            for data in sentiment_data:
                for topic in data.topics:
                    if topic not in topic_sentiment:
                        topic_sentiment[topic] = []
                        topic_counts[topic] = 0
                    
                    topic_sentiment[topic].append(data.sentiment_score)
                    topic_counts[topic] += 1
            
            # Calculate average sentiment per topic
            for topic in topic_sentiment:
                topic_sentiment[topic] = np.mean(topic_sentiment[topic])
            
            # Get top topics by volume
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_topics = [topic for topic, count in top_topics]
            
            return {
                'top_topics': top_topics,
                'topic_sentiment': topic_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error calculating topic analysis: {e}")
            return {'top_topics': [], 'topic_sentiment': {}}
    
    def _calculate_entity_analysis(self, sentiment_data: List[SentimentData]) -> Dict[str, Any]:
        """Calculate entity-based sentiment analysis"""
        try:
            entity_sentiment = {}
            entity_counts = {}
            
            for data in sentiment_data:
                for entity in data.entities:
                    if entity not in entity_sentiment:
                        entity_sentiment[entity] = []
                        entity_counts[entity] = 0
                    
                    entity_sentiment[entity].append(data.sentiment_score)
                    entity_counts[entity] += 1
            
            # Calculate average sentiment per entity
            for entity in entity_sentiment:
                entity_sentiment[entity] = np.mean(entity_sentiment[entity])
            
            # Get top entities by volume
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_entities = [entity for entity, count in top_entities]
            
            return {
                'top_entities': top_entities,
                'entity_sentiment': entity_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error calculating entity analysis: {e}")
            return {'top_entities': [], 'entity_sentiment': {}}
    
    async def _calculate_market_correlations(self, symbol: str, analysis: SentimentAnalysis) -> SentimentAnalysis:
        """Calculate market correlations"""
        try:
            # Get historical price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if hist.empty:
                return analysis
            
            # Get sentiment history for correlation
            sentiment_history = self.sentiment_history.get(symbol, [])
            
            if len(sentiment_history) >= 5:
                # Calculate correlations
                sentiment_scores = [s.overall_sentiment for s in sentiment_history[-len(hist):]]
                price_changes = hist['Close'].pct_change().dropna()
                
                if len(sentiment_scores) == len(price_changes):
                    price_correlation = np.corrcoef(sentiment_scores, price_changes)[0, 1]
                    volume_correlation = np.corrcoef(sentiment_scores, hist['Volume'].pct_change().dropna())[0, 1]
                    
                    analysis.price_correlation = price_correlation if not np.isnan(price_correlation) else 0
                    analysis.volume_correlation = volume_correlation if not np.isnan(volume_correlation) else 0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error calculating market correlations: {e}")
            return analysis
    
    async def _generate_predictive_signals(self, analysis: SentimentAnalysis) -> SentimentAnalysis:
        """Generate predictive sentiment signals"""
        try:
            bullish_signals = []
            bearish_signals = []
            
            # Sentiment-based signals
            if analysis.overall_sentiment > 0.3:
                bullish_signals.append("Strong positive sentiment")
            elif analysis.overall_sentiment < -0.3:
                bearish_signals.append("Strong negative sentiment")
            
            # Momentum-based signals
            if analysis.sentiment_momentum > 0.2:
                bullish_signals.append("Improving sentiment momentum")
            elif analysis.sentiment_momentum < -0.2:
                bearish_signals.append("Deteriorating sentiment momentum")
            
            # Volume-based signals
            if analysis.positive_volume > analysis.negative_volume * 2:
                bullish_signals.append("High positive sentiment volume")
            elif analysis.negative_volume > analysis.positive_volume * 2:
                bearish_signals.append("High negative sentiment volume")
            
            # Topic-based signals
            for topic, sentiment in analysis.topic_sentiment.items():
                if sentiment > 0.2:
                    bullish_signals.append(f"Positive {topic} sentiment")
                elif sentiment < -0.2:
                    bearish_signals.append(f"Negative {topic} sentiment")
            
            # Market correlation signals
            if analysis.price_correlation > 0.5:
                bullish_signals.append("Strong positive sentiment-price correlation")
            elif analysis.price_correlation < -0.5:
                bearish_signals.append("Strong negative sentiment-price correlation")
            
            analysis.bullish_signals = bullish_signals
            analysis.bearish_signals = bearish_signals
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating predictive signals: {e}")
            return analysis
    
    def _update_sentiment_cache(self, symbol: str, analysis: SentimentAnalysis):
        """Update sentiment cache and history"""
        try:
            # Update cache
            self.sentiment_cache[symbol] = analysis
            
            # Update history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            
            self.sentiment_history[symbol].append(analysis)
            
            # Keep only last 30 days of history
            cutoff_date = datetime.now() - timedelta(days=30)
            self.sentiment_history[symbol] = [
                s for s in self.sentiment_history[symbol]
                if s.timestamp > cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Error updating sentiment cache: {e}")
    
    def _create_empty_analysis(self, symbol: str) -> SentimentAnalysis:
        """Create empty sentiment analysis"""
        return SentimentAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_sentiment=0,
            sentiment_type=SentimentType.NEUTRAL,
            confidence=0,
            news_sentiment=0,
            social_sentiment=0,
            reddit_sentiment=0,
            twitter_sentiment=0,
            sentiment_trend=0,
            sentiment_momentum=0,
            sentiment_volatility=0,
            sentiment_volume=0,
            positive_volume=0,
            negative_volume=0,
            neutral_volume=0,
            top_topics=[],
            topic_sentiment={},
            top_entities=[],
            entity_sentiment={},
            price_correlation=0,
            volume_correlation=0,
            bullish_signals=[],
            bearish_signals=[],
            sentiment_data=[],
            analysis_duration=0,
            data_sources=[],
            model_confidence=0
        )
    
    def get_sentiment_cache(self, symbol: str) -> Optional[SentimentAnalysis]:
        """Get cached sentiment analysis"""
        return self.sentiment_cache.get(symbol)
    
    def get_sentiment_history(self, symbol: str) -> List[SentimentAnalysis]:
        """Get sentiment history for symbol"""
        return self.sentiment_history.get(symbol, [])
    
    def get_all_sentiment_data(self) -> Dict[str, SentimentAnalysis]:
        """Get all cached sentiment data"""
        return self.sentiment_cache.copy()