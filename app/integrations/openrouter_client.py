#!/usr/bin/env python3
"""
OpenRouter LLM Client
Provides access to various LLM models through OpenRouter API
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import aiohttp
import json
from datetime import datetime
import base64
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    cost: float
    timestamp: datetime
    metadata: Dict = None

class OpenRouterClient:
    """
    OpenRouter API client for LLM integration
    Supports multiple models with trading-specific functionality
    """
    
    def __init__(self, api_key: str, app_name: str = "TradingBot"):
        self.api_key = api_key
        self.app_name = app_name
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = None
        
        # Model configurations
        self.models = {
            'gpt-4': {
                'id': 'openai/gpt-4',
                'context_length': 8192,
                'cost_per_token': 0.00003,
                'capabilities': ['analysis', 'strategy', 'execution']
            },
            'gpt-4-turbo': {
                'id': 'openai/gpt-4-turbo',
                'context_length': 128000,
                'cost_per_token': 0.00001,
                'capabilities': ['analysis', 'strategy', 'execution', 'document']
            },
            'claude-3-opus': {
                'id': 'anthropic/claude-3-opus',
                'context_length': 200000,
                'cost_per_token': 0.000015,
                'capabilities': ['analysis', 'strategy', 'execution', 'document', 'reasoning']
            },
            'claude-3-sonnet': {
                'id': 'anthropic/claude-3-sonnet',
                'context_length': 200000,
                'cost_per_token': 0.000003,
                'capabilities': ['analysis', 'strategy', 'execution', 'document']
            },
            'gemini-pro': {
                'id': 'google/gemini-pro',
                'context_length': 32768,
                'cost_per_token': 0.000001,
                'capabilities': ['analysis', 'strategy']
            }
        }
        
        self.default_model = 'claude-3-sonnet'
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat_completion(self, 
                            messages: List[ChatMessage], 
                            model: str = None,
                            temperature: float = 0.7,
                            max_tokens: int = 2048,
                            tools: Optional[List[Dict]] = None) -> Optional[LLMResponse]:
        """Send chat completion request to OpenRouter"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            model_id = self.models.get(model or self.default_model, {}).get('id', self.models[self.default_model]['id'])
            
            # Prepare messages
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    'role': msg.role,
                    'content': msg.content
                })
            
            payload = {
                'model': model_id,
                'messages': formatted_messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'stream': False
            }
            
            if tools:
                payload['tools'] = tools
                payload['tool_choice'] = 'auto'
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://trading-bot.com',
                'X-Title': self.app_name
            }
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    choice = data['choices'][0]
                    usage = data.get('usage', {})
                    
                    return LLMResponse(
                        content=choice['message']['content'],
                        model=model_id,
                        tokens_used=usage.get('total_tokens', 0),
                        cost=self._calculate_cost(usage.get('total_tokens', 0), model or self.default_model),
                        timestamp=datetime.now(),
                        metadata={
                            'finish_reason': choice.get('finish_reason'),
                            'usage': usage,
                            'tool_calls': choice['message'].get('tool_calls')
                        }
                    )
                else:
                    error_data = await response.json()
                    logger.error(f"OpenRouter API error: {error_data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            return None
    
    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost based on token usage"""
        model_config = self.models.get(model, self.models[self.default_model])
        return tokens * model_config['cost_per_token']
    
    async def analyze_market_data(self, market_data: Dict, context: str = "") -> Optional[str]:
        """Analyze market data using LLM"""
        try:
            system_prompt = """You are an expert financial analyst and trader. Analyze the provided market data and provide insights including:
1. Current market trends and patterns
2. Key support and resistance levels
3. Technical indicators analysis
4. Market sentiment assessment
5. Trading opportunities and risks
6. Recommended actions

Be specific, actionable, and include reasoning for your analysis."""

            user_content = f"""Market Data Analysis Request:
{context}

Market Data:
{json.dumps(market_data, indent=2, default=str)}

Please provide a comprehensive analysis of this market data."""

            messages = [
                ChatMessage(role='system', content=system_prompt),
                ChatMessage(role='user', content=user_content)
            ]
            
            response = await self.chat_completion(messages, model='claude-3-sonnet')
            return response.content if response else None
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return None
    
    async def build_trading_strategy(self, requirements: Dict) -> Optional[Dict]:
        """Build a trading strategy based on requirements"""
        try:
            system_prompt = """You are an expert quantitative trader and strategy developer. Create comprehensive trading strategies based on user requirements. Return your response as a JSON object with the following structure:

{
    "strategy_name": "string",
    "description": "string", 
    "parameters": {
        "timeframe": "string",
        "risk_level": "low|medium|high",
        "max_drawdown": "percentage",
        "target_return": "percentage",
        "position_size": "percentage",
        "stop_loss": "percentage",
        "take_profit": "percentage"
    },
    "entry_conditions": ["list of conditions"],
    "exit_conditions": ["list of conditions"],
    "risk_management": {
        "max_positions": "number",
        "correlation_limit": "percentage",
        "var_limit": "percentage"
    },
    "indicators": ["list of technical indicators"],
    "backtesting_period": "string",
    "implementation_notes": "string"
}"""

            user_content = f"""Create a trading strategy with these requirements:
{json.dumps(requirements, indent=2)}

Ensure the strategy is practical, well-defined, and includes proper risk management."""

            messages = [
                ChatMessage(role='system', content=system_prompt),
                ChatMessage(role='user', content=user_content)
            ]
            
            response = await self.chat_completion(messages, model='gpt-4-turbo', temperature=0.3)
            
            if response:
                try:
                    # Parse JSON response
                    strategy_data = json.loads(response.content)
                    return strategy_data
                except json.JSONDecodeError:
                    # If not valid JSON, return as text
                    return {'description': response.content}
            
            return None
            
        except Exception as e:
            logger.error(f"Strategy building error: {e}")
            return None
    
    async def optimize_strategy_parameters(self, strategy: Dict, performance_data: Dict) -> Optional[Dict]:
        """Optimize strategy parameters based on performance"""
        try:
            system_prompt = """You are a quantitative analyst specializing in strategy optimization. Analyze the strategy performance and suggest parameter improvements. Return optimized parameters as JSON."""

            user_content = f"""Strategy to optimize:
{json.dumps(strategy, indent=2)}

Performance Data:
{json.dumps(performance_data, indent=2)}

Analyze the performance and suggest optimized parameters to improve risk-adjusted returns."""

            messages = [
                ChatMessage(role='system', content=system_prompt),
                ChatMessage(role='user', content=user_content)
            ]
            
            response = await self.chat_completion(messages, model='claude-3-opus', temperature=0.2)
            
            if response:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {'recommendations': response.content}
            
            return None
            
        except Exception as e:
            logger.error(f"Parameter optimization error: {e}")
            return None
    
    async def forecast_market_direction(self, historical_data: List[Dict], 
                                      external_factors: Dict = None) -> Optional[Dict]:
        """Forecast market direction using LLM analysis"""
        try:
            system_prompt = """You are a senior market analyst with expertise in forecasting. Analyze historical data and external factors to predict market direction. Provide:
1. Short-term forecast (1-7 days)
2. Medium-term forecast (1-4 weeks)
3. Long-term forecast (1-3 months)
4. Confidence levels for each forecast
5. Key factors influencing the forecast
6. Potential risks and scenarios

Return as JSON with confidence scores (0-100)."""

            user_content = f"""Historical Market Data (last {len(historical_data)} periods):
{json.dumps(historical_data[-50:], indent=2, default=str)}  # Last 50 data points

External Factors:
{json.dumps(external_factors or {}, indent=2, default=str)}

Provide market direction forecast with reasoning."""

            messages = [
                ChatMessage(role='system', content=system_prompt),
                ChatMessage(role='user', content=user_content)
            ]
            
            response = await self.chat_completion(messages, model='gpt-4-turbo')
            
            if response:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {'forecast': response.content}
            
            return None
            
        except Exception as e:
            logger.error(f"Market forecasting error: {e}")
            return None
    
    async def analyze_trading_plan(self, plan_text: str) -> Optional[Dict]:
        """Analyze uploaded trading plan document"""
        try:
            system_prompt = """You are a professional trading plan analyst. Review the trading plan and provide:
1. Strengths and weaknesses assessment
2. Risk analysis
3. Feasibility evaluation
4. Improvement suggestions
5. Implementation roadmap
6. Expected performance metrics

Return as structured JSON analysis."""

            user_content = f"""Trading Plan to Analyze:

{plan_text}

Provide comprehensive analysis and recommendations."""

            messages = [
                ChatMessage(role='system', content=system_prompt),
                ChatMessage(role='user', content=user_content)
            ]
            
            response = await self.chat_completion(messages, model='claude-3-opus')
            
            if response:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {'analysis': response.content}
            
            return None
            
        except Exception as e:
            logger.error(f"Trading plan analysis error: {e}")
            return None
    
    async def execute_trade_command(self, command: str, context: Dict) -> Optional[Dict]:
        """Parse and execute trade commands from natural language"""
        try:
            system_prompt = """You are a trading execution assistant. Parse natural language trading commands and return structured execution instructions.

Return JSON format:
{
    "action": "buy|sell|close|modify",
    "symbol": "trading pair",
    "quantity": "amount or percentage",
    "price": "limit price or 'market'",
    "stop_loss": "stop loss level",
    "take_profit": "take profit level",
    "urgency": "immediate|normal|delayed",
    "risk_check": "approved|needs_review|rejected",
    "reasoning": "explanation"
}"""

            user_content = f"""Trading Command: {command}

Context:
- Available Balance: {context.get('balance', 'Unknown')}
- Current Positions: {json.dumps(context.get('positions', []), default=str)}
- Active Orders: {json.dumps(context.get('orders', []), default=str)}
- Risk Limits: {json.dumps(context.get('risk_limits', {}), default=str)}

Parse this command and provide execution instructions."""

            messages = [
                ChatMessage(role='system', content=system_prompt),
                ChatMessage(role='user', content=user_content)
            ]
            
            response = await self.chat_completion(messages, model='gpt-4', temperature=0.1)
            
            if response:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {'error': 'Could not parse command', 'response': response.content}
            
            return None
            
        except Exception as e:
            logger.error(f"Trade command execution error: {e}")
            return None
    
    async def generate_trading_report(self, account_data: Dict, period: str = "daily") -> Optional[str]:
        """Generate comprehensive trading report"""
        try:
            system_prompt = f"""You are a professional trading report analyst. Generate a comprehensive {period} trading report with:
1. Performance summary
2. Trade analysis
3. Risk assessment
4. Market overview
5. Strategy performance
6. Recommendations for improvement

Format as a professional report with clear sections and insights."""

            user_content = f"""Account Data for {period.title()} Report:
{json.dumps(account_data, indent=2, default=str)}

Generate a professional trading report."""

            messages = [
                ChatMessage(role='system', content=system_prompt),
                ChatMessage(role='user', content=user_content)
            ]
            
            response = await self.chat_completion(messages, model='claude-3-sonnet')
            return response.content if response else None
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return None
    
    async def get_model_info(self) -> Dict:
        """Get available models and their capabilities"""
        return {
            'available_models': self.models,
            'default_model': self.default_model,
            'capabilities': [
                'market_analysis',
                'strategy_building',
                'parameter_optimization', 
                'forecasting',
                'plan_analysis',
                'trade_execution',
                'report_generation'
            ]
        }
    
    def close(self):
        """Close the session"""
        if self.session:
            asyncio.create_task(self.session.close())