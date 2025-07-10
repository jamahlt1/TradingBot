import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketAnalysis:
    """Market analysis result from LLM"""
    sentiment: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    key_factors: List[str]
    risk_assessment: str
    recommendations: List[str]
    technical_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    market_timing: str
    position_sizing: Dict[str, float]

class LLMAnalyzer:
    """
    Advanced LLM Trading Analysis
    - OpenRouter integration for multiple models
    - DeepSeek integration for specialized analysis
    - Market sentiment analysis
    - Strategy optimization
    - Risk assessment
    """
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 deepseek_api_key: Optional[str] = None):
        
        self.openrouter_api_key = openrouter_api_key
        self.deepseek_api_key = deepseek_api_key
        self.openrouter_url = "https://openrouter.ai/api/v1"
        self.deepseek_url = "https://api.deepseek.com/v1"
        
        self.session = None
        self.analysis_cache = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_openrouter_request(self, 
                                     model: str, 
                                     messages: List[Dict], 
                                     temperature: float = 0.7) -> Dict:
        """Make request to OpenRouter API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            'Authorization': f'Bearer {self.openrouter_api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://trading-bot-platform.com',
            'X-Title': 'Trading Bot Platform'
        }
        
        payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': 2000
        }
        
        try:
            async with self.session.post(
                f"{self.openrouter_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    logger.error(f"OpenRouter API error: {response.status}")
                    return ""
        except Exception as e:
            logger.error(f"OpenRouter request error: {e}")
            return ""
    
    async def _make_deepseek_request(self, 
                                    messages: List[Dict], 
                                    temperature: float = 0.7) -> Dict:
        """Make request to DeepSeek API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            'Authorization': f'Bearer {self.deepseek_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'deepseek-chat',
            'messages': messages,
            'temperature': temperature,
            'max_tokens': 2000
        }
        
        try:
            async with self.session.post(
                f"{self.deepseek_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    logger.error(f"DeepSeek API error: {response.status}")
                    return ""
        except Exception as e:
            logger.error(f"DeepSeek request error: {e}")
            return ""
    
    async def analyze_market_sentiment(self, 
                                     symbol: str, 
                                     market_data: pd.DataFrame,
                                     news_data: List[Dict] = None) -> MarketAnalysis:
        """Analyze market sentiment using LLM"""
        try:
            # Prepare market data summary
            data_summary = self._prepare_market_data_summary(symbol, market_data)
            
            # Prepare news summary
            news_summary = self._prepare_news_summary(news_data) if news_data else ""
            
            # Create analysis prompt
            prompt = f"""
            Analyze the market sentiment for {symbol} based on the following data:
            
            Market Data Summary:
            {data_summary}
            
            Recent News:
            {news_summary}
            
            Please provide a comprehensive analysis including:
            1. Overall sentiment (bullish/bearish/neutral)
            2. Confidence level (0-1)
            3. Key factors influencing the sentiment
            4. Risk assessment
            5. Trading recommendations
            6. Technical analysis insights
            7. Fundamental analysis insights
            8. Market timing assessment
            9. Position sizing recommendations
            
            Format your response as JSON with the following structure:
            {{
                "sentiment": "bullish/bearish/neutral",
                "confidence": 0.85,
                "key_factors": ["factor1", "factor2"],
                "risk_assessment": "low/medium/high",
                "recommendations": ["rec1", "rec2"],
                "technical_analysis": {{"trend": "upward", "support": 100, "resistance": 110}},
                "fundamental_analysis": {{"valuation": "fair", "growth": "positive"}},
                "market_timing": "good/bad/neutral",
                "position_sizing": {{"suggested_size": 0.1, "max_size": 0.2}}
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert financial analyst specializing in market sentiment analysis and trading recommendations."},
                {"role": "user", "content": prompt}
            ]
            
            # Get analysis from multiple models
            analyses = []
            
            # OpenRouter models
            openrouter_models = [
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4-turbo",
                "meta-llama/llama-3.1-8b-instruct"
            ]
            
            for model in openrouter_models:
                if self.openrouter_api_key:
                    response = await self._make_openrouter_request(model, messages)
                    if response:
                        try:
                            analysis = json.loads(response)
                            analyses.append(analysis)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON from {model}")
            
            # DeepSeek analysis
            if self.deepseek_api_key:
                deepseek_response = await self._make_deepseek_request(messages)
                if deepseek_response:
                    try:
                        analysis = json.loads(deepseek_response)
                        analyses.append(analysis)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse JSON from DeepSeek")
            
            # Combine analyses
            if analyses:
                combined_analysis = self._combine_analyses(analyses)
                return MarketAnalysis(**combined_analysis)
            else:
                # Fallback analysis
                return self._create_fallback_analysis(symbol, market_data)
                
        except Exception as e:
            logger.error(f"Error in market sentiment analysis: {e}")
            return self._create_fallback_analysis(symbol, market_data)
    
    async def optimize_strategy_parameters(self, 
                                        strategy_name: str,
                                        historical_performance: Dict,
                                        market_conditions: Dict) -> Dict[str, Any]:
        """Optimize strategy parameters using LLM"""
        try:
            prompt = f"""
            Optimize the parameters for the {strategy_name} trading strategy based on:
            
            Historical Performance:
            {json.dumps(historical_performance, indent=2)}
            
            Current Market Conditions:
            {json.dumps(market_conditions, indent=2)}
            
            Please suggest optimized parameters that would improve performance while managing risk.
            Consider:
            1. Market volatility adjustments
            2. Risk management improvements
            3. Entry/exit timing optimization
            4. Position sizing adjustments
            
            Format your response as JSON with the following structure:
            {{
                "optimized_parameters": {{"param1": value1, "param2": value2}},
                "reasoning": "explanation for changes",
                "expected_improvement": "description of expected improvements",
                "risk_adjustments": "risk management changes"
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert quantitative trader specializing in strategy optimization and risk management."},
                {"role": "user", "content": prompt}
            ]
            
            # Get optimization from multiple models
            optimizations = []
            
            if self.openrouter_api_key:
                for model in ["anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo"]:
                    response = await self._make_openrouter_request(model, messages)
                    if response:
                        try:
                            optimization = json.loads(response)
                            optimizations.append(optimization)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse optimization from {model}")
            
            if self.deepseek_api_key:
                deepseek_response = await self._make_deepseek_request(messages)
                if deepseek_response:
                    try:
                        optimization = json.loads(deepseek_response)
                        optimizations.append(optimization)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse optimization from DeepSeek")
            
            # Combine optimizations
            if optimizations:
                return self._combine_optimizations(optimizations)
            else:
                return self._create_fallback_optimization(strategy_name)
                
        except Exception as e:
            logger.error(f"Error in strategy optimization: {e}")
            return self._create_fallback_optimization(strategy_name)
    
    async def compare_trading_opportunities(self, 
                                         opportunities: List[Dict]) -> Dict[str, Any]:
        """Compare multiple trading opportunities using LLM"""
        try:
            prompt = f"""
            Compare the following trading opportunities and rank them by potential:
            
            Opportunities:
            {json.dumps(opportunities, indent=2)}
            
            Please analyze each opportunity considering:
            1. Risk-reward ratio
            2. Market conditions
            3. Technical indicators
            4. Fundamental factors
            5. Liquidity and execution risk
            
            Rank them from best to worst and provide reasoning.
            
            Format your response as JSON:
            {{
                "ranked_opportunities": [
                    {{"rank": 1, "opportunity_id": "id", "reasoning": "explanation"}}
                ],
                "overall_assessment": "summary of opportunities",
                "risk_warnings": ["warning1", "warning2"],
                "recommendations": ["rec1", "rec2"]
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert portfolio manager specializing in opportunity analysis and risk assessment."},
                {"role": "user", "content": prompt}
            ]
            
            if self.openrouter_api_key:
                response = await self._make_openrouter_request("anthropic/claude-3.5-sonnet", messages)
                if response:
                    try:
                        return json.loads(response)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse opportunity comparison")
            
            return self._create_fallback_comparison(opportunities)
            
        except Exception as e:
            logger.error(f"Error in opportunity comparison: {e}")
            return self._create_fallback_comparison(opportunities)
    
    async def analyze_risk_factors(self, 
                                 portfolio: Dict,
                                 market_data: Dict) -> Dict[str, Any]:
        """Analyze portfolio risk factors using LLM"""
        try:
            prompt = f"""
            Analyze the risk factors for the following portfolio:
            
            Portfolio:
            {json.dumps(portfolio, indent=2)}
            
            Market Data:
            {json.dumps(market_data, indent=2)}
            
            Please identify:
            1. Concentration risks
            2. Correlation risks
            3. Market timing risks
            4. Liquidity risks
            5. Volatility risks
            6. Recommendations for risk mitigation
            
            Format your response as JSON:
            {{
                "risk_factors": [
                    {{"type": "concentration", "severity": "high", "description": "..."}}
                ],
                "overall_risk_score": 0.75,
                "risk_level": "medium",
                "mitigation_strategies": ["strategy1", "strategy2"],
                "recommendations": ["rec1", "rec2"]
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are an expert risk manager specializing in portfolio risk analysis and mitigation strategies."},
                {"role": "user", "content": prompt}
            ]
            
            if self.deepseek_api_key:
                response = await self._make_deepseek_request(messages)
                if response:
                    try:
                        return json.loads(response)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse risk analysis")
            
            return self._create_fallback_risk_analysis(portfolio)
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return self._create_fallback_risk_analysis(portfolio)
    
    def _prepare_market_data_summary(self, symbol: str, market_data: pd.DataFrame) -> str:
        """Prepare market data summary for LLM analysis"""
        try:
            if market_data.empty:
                return "No market data available"
            
            latest = market_data.iloc[-1]
            recent_data = market_data.tail(20)
            
            summary = f"""
            Symbol: {symbol}
            Current Price: ${latest['close']:.2f}
            24h Change: {((latest['close'] - market_data.iloc[-2]['close']) / market_data.iloc[-2]['close'] * 100):.2f}%
            24h High: ${latest['high']:.2f}
            24h Low: ${latest['low']:.2f}
            Volume: {latest['volume']:,.0f}
            
            Recent Performance (20 periods):
            - Price Range: ${recent_data['low'].min():.2f} - ${recent_data['high'].max():.2f}
            - Average Volume: {recent_data['volume'].mean():,.0f}
            - Volatility: {recent_data['close'].pct_change().std() * 100:.2f}%
            """
            
            return summary
            
        except Exception as e:
            logger.error(f"Error preparing market data summary: {e}")
            return "Error preparing market data"
    
    def _prepare_news_summary(self, news_data: List[Dict]) -> str:
        """Prepare news summary for LLM analysis"""
        try:
            if not news_data:
                return "No recent news available"
            
            summary = "Recent News:\n"
            for i, news in enumerate(news_data[:5]):  # Top 5 news items
                summary += f"{i+1}. {news.get('title', 'No title')}\n"
                summary += f"   Sentiment: {news.get('sentiment', 'neutral')}\n"
                summary += f"   Impact: {news.get('impact', 'low')}\n\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error preparing news summary: {e}")
            return "Error preparing news summary"
    
    def _combine_analyses(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Combine multiple LLM analyses"""
        try:
            # Average sentiment scores
            sentiment_scores = {
                'bullish': 1,
                'neutral': 0,
                'bearish': -1
            }
            
            total_score = 0
            total_confidence = 0
            all_factors = []
            all_recommendations = []
            
            for analysis in analyses:
                sentiment = analysis.get('sentiment', 'neutral')
                total_score += sentiment_scores.get(sentiment, 0)
                total_confidence += analysis.get('confidence', 0.5)
                all_factors.extend(analysis.get('key_factors', []))
                all_recommendations.extend(analysis.get('recommendations', []))
            
            avg_score = total_score / len(analyses)
            avg_confidence = total_confidence / len(analyses)
            
            # Determine final sentiment
            if avg_score > 0.3:
                final_sentiment = 'bullish'
            elif avg_score < -0.3:
                final_sentiment = 'bearish'
            else:
                final_sentiment = 'neutral'
            
            return {
                'sentiment': final_sentiment,
                'confidence': avg_confidence,
                'key_factors': list(set(all_factors)),
                'risk_assessment': analyses[0].get('risk_assessment', 'medium'),
                'recommendations': list(set(all_recommendations)),
                'technical_analysis': analyses[0].get('technical_analysis', {}),
                'fundamental_analysis': analyses[0].get('fundamental_analysis', {}),
                'market_timing': analyses[0].get('market_timing', 'neutral'),
                'position_sizing': analyses[0].get('position_sizing', {})
            }
            
        except Exception as e:
            logger.error(f"Error combining analyses: {e}")
            return self._create_fallback_analysis("", pd.DataFrame())
    
    def _combine_optimizations(self, optimizations: List[Dict]) -> Dict[str, Any]:
        """Combine multiple optimization suggestions"""
        try:
            # Take the first optimization as base and combine reasoning
            base_optimization = optimizations[0]
            
            combined_reasoning = "Combined analysis from multiple models:\n"
            for i, opt in enumerate(optimizations):
                combined_reasoning += f"Model {i+1}: {opt.get('reasoning', 'No reasoning provided')}\n"
            
            base_optimization['reasoning'] = combined_reasoning
            
            return base_optimization
            
        except Exception as e:
            logger.error(f"Error combining optimizations: {e}")
            return self._create_fallback_optimization("")
    
    def _create_fallback_analysis(self, symbol: str, market_data: pd.DataFrame) -> MarketAnalysis:
        """Create fallback analysis when LLM fails"""
        return MarketAnalysis(
            sentiment='neutral',
            confidence=0.5,
            key_factors=['Limited data available'],
            risk_assessment='medium',
            recommendations=['Monitor market conditions'],
            technical_analysis={'trend': 'neutral'},
            fundamental_analysis={'valuation': 'unknown'},
            market_timing='neutral',
            position_sizing={'suggested_size': 0.05, 'max_size': 0.1}
        )
    
    def _create_fallback_optimization(self, strategy_name: str) -> Dict[str, Any]:
        """Create fallback optimization when LLM fails"""
        return {
            'optimized_parameters': {},
            'reasoning': 'LLM analysis unavailable',
            'expected_improvement': 'No changes recommended',
            'risk_adjustments': 'Maintain current risk settings'
        }
    
    def _create_fallback_comparison(self, opportunities: List[Dict]) -> Dict[str, Any]:
        """Create fallback opportunity comparison"""
        return {
            'ranked_opportunities': [
                {'rank': i+1, 'opportunity_id': f"opp_{i}", 'reasoning': 'Limited analysis available'}
                for i in range(len(opportunities))
            ],
            'overall_assessment': 'Limited analysis available',
            'risk_warnings': ['Use caution with limited data'],
            'recommendations': ['Monitor market conditions']
        }
    
    def _create_fallback_risk_analysis(self, portfolio: Dict) -> Dict[str, Any]:
        """Create fallback risk analysis"""
        return {
            'risk_factors': [
                {'type': 'data_limited', 'severity': 'medium', 'description': 'Limited data for analysis'}
            ],
            'overall_risk_score': 0.5,
            'risk_level': 'medium',
            'mitigation_strategies': ['Monitor positions closely'],
            'recommendations': ['Use conservative position sizing']
        }