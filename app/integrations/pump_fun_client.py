import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TokenInfo:
    """Token information from pump.fun"""
    address: str
    name: str
    symbol: str
    price: float
    market_cap: float
    volume_24h: float
    holders: int
    created_at: datetime
    is_graduated: bool
    graduation_date: Optional[datetime] = None
    risk_score: float = 0.0

class PumpFunClient:
    """
    Pump.fun Integration Client
    - Token discovery and tracking
    - Price monitoring
    - Trading execution
    - Risk assessment
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.pump.fun"
        self.session = None
        self.tokens_cache = {}
        self.price_cache = {}
        self.last_update = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request to pump.fun"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            'User-Agent': 'TradingBot/1.0',
            'Accept': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API request failed: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {}
    
    async def get_trending_tokens(self, limit: int = 50) -> List[TokenInfo]:
        """Get trending tokens from pump.fun"""
        try:
            data = await self._make_request('/v1/tokens/trending', {'limit': limit})
            
            tokens = []
            for token_data in data.get('tokens', []):
                token = TokenInfo(
                    address=token_data.get('address'),
                    name=token_data.get('name'),
                    symbol=token_data.get('symbol'),
                    price=float(token_data.get('price', 0)),
                    market_cap=float(token_data.get('marketCap', 0)),
                    volume_24h=float(token_data.get('volume24h', 0)),
                    holders=int(token_data.get('holders', 0)),
                    created_at=datetime.fromisoformat(token_data.get('createdAt')),
                    is_graduated=token_data.get('isGraduated', False),
                    graduation_date=datetime.fromisoformat(token_data.get('graduationDate')) if token_data.get('graduationDate') else None,
                    risk_score=self._calculate_risk_score(token_data)
                )
                tokens.append(token)
            
            return tokens
        except Exception as e:
            logger.error(f"Error fetching trending tokens: {e}")
            return []
    
    async def get_token_info(self, token_address: str) -> Optional[TokenInfo]:
        """Get detailed token information"""
        try:
            data = await self._make_request(f'/v1/tokens/{token_address}')
            
            if not data:
                return None
            
            return TokenInfo(
                address=data.get('address'),
                name=data.get('name'),
                symbol=data.get('symbol'),
                price=float(data.get('price', 0)),
                market_cap=float(data.get('marketCap', 0)),
                volume_24h=float(data.get('volume24h', 0)),
                holders=int(data.get('holders', 0)),
                created_at=datetime.fromisoformat(data.get('createdAt')),
                is_graduated=data.get('isGraduated', False),
                graduation_date=datetime.fromisoformat(data.get('graduationDate')) if data.get('graduationDate') else None,
                risk_score=self._calculate_risk_score(data)
            )
        except Exception as e:
            logger.error(f"Error fetching token info: {e}")
            return None
    
    async def get_token_price_history(self, token_address: str, timeframe: str = '1h', limit: int = 100) -> List[Dict]:
        """Get token price history"""
        try:
            params = {
                'timeframe': timeframe,
                'limit': limit
            }
            
            data = await self._make_request(f'/v1/tokens/{token_address}/price-history', params)
            
            return data.get('prices', [])
        except Exception as e:
            logger.error(f"Error fetching price history: {e}")
            return []
    
    async def get_new_tokens(self, hours: int = 24) -> List[TokenInfo]:
        """Get newly created tokens"""
        try:
            since = datetime.now() - timedelta(hours=hours)
            params = {
                'since': since.isoformat(),
                'limit': 100
            }
            
            data = await self._make_request('/v1/tokens/new', params)
            
            tokens = []
            for token_data in data.get('tokens', []):
                token = TokenInfo(
                    address=token_data.get('address'),
                    name=token_data.get('name'),
                    symbol=token_data.get('symbol'),
                    price=float(token_data.get('price', 0)),
                    market_cap=float(token_data.get('marketCap', 0)),
                    volume_24h=float(token_data.get('volume24h', 0)),
                    holders=int(token_data.get('holders', 0)),
                    created_at=datetime.fromisoformat(token_data.get('createdAt')),
                    is_graduated=False,
                    risk_score=self._calculate_risk_score(token_data)
                )
                tokens.append(token)
            
            return tokens
        except Exception as e:
            logger.error(f"Error fetching new tokens: {e}")
            return []
    
    async def get_graduated_tokens(self) -> List[TokenInfo]:
        """Get graduated tokens"""
        try:
            data = await self._make_request('/v1/tokens/graduated')
            
            tokens = []
            for token_data in data.get('tokens', []):
                token = TokenInfo(
                    address=token_data.get('address'),
                    name=token_data.get('name'),
                    symbol=token_data.get('symbol'),
                    price=float(token_data.get('price', 0)),
                    market_cap=float(token_data.get('marketCap', 0)),
                    volume_24h=float(token_data.get('volume24h', 0)),
                    holders=int(token_data.get('holders', 0)),
                    created_at=datetime.fromisoformat(token_data.get('createdAt')),
                    is_graduated=True,
                    graduation_date=datetime.fromisoformat(token_data.get('graduationDate')),
                    risk_score=self._calculate_risk_score(token_data)
                )
                tokens.append(token)
            
            return tokens
        except Exception as e:
            logger.error(f"Error fetching graduated tokens: {e}")
            return []
    
    def _calculate_risk_score(self, token_data: Dict) -> float:
        """Calculate risk score for a token"""
        risk_score = 0.0
        
        # Market cap risk
        market_cap = float(token_data.get('marketCap', 0))
        if market_cap < 10000:  # Very small market cap
            risk_score += 0.4
        elif market_cap < 100000:
            risk_score += 0.2
        
        # Holder count risk
        holders = int(token_data.get('holders', 0))
        if holders < 100:
            risk_score += 0.3
        elif holders < 500:
            risk_score += 0.15
        
        # Volume risk
        volume_24h = float(token_data.get('volume24h', 0))
        if volume_24h < 1000:
            risk_score += 0.2
        elif volume_24h < 10000:
            risk_score += 0.1
        
        # Age risk (newer tokens are riskier)
        created_at = datetime.fromisoformat(token_data.get('createdAt'))
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        if age_hours < 24:
            risk_score += 0.3
        elif age_hours < 168:  # 1 week
            risk_score += 0.15
        
        # Graduation status
        if token_data.get('isGraduated', False):
            risk_score -= 0.2  # Graduated tokens are less risky
        
        return min(max(risk_score, 0.0), 1.0)
    
    async def monitor_token(self, token_address: str, callback: callable):
        """Monitor token price changes"""
        last_price = None
        
        while True:
            try:
                token_info = await self.get_token_info(token_address)
                if token_info:
                    current_price = token_info.price
                    
                    if last_price is not None and current_price != last_price:
                        price_change = ((current_price - last_price) / last_price) * 100
                        
                        await callback({
                            'token_address': token_address,
                            'symbol': token_info.symbol,
                            'price': current_price,
                            'price_change': price_change,
                            'timestamp': datetime.now(),
                            'market_cap': token_info.market_cap,
                            'volume_24h': token_info.volume_24h,
                            'holders': token_info.holders
                        })
                    
                    last_price = current_price
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring token {token_address}: {e}")
                await asyncio.sleep(60)
    
    async def get_trading_opportunities(self) -> List[Dict]:
        """Identify trading opportunities"""
        opportunities = []
        
        try:
            # Get trending tokens
            trending_tokens = await self.get_trending_tokens(20)
            
            for token in trending_tokens:
                # Analyze for opportunities
                opportunity = await self._analyze_token_opportunity(token)
                if opportunity:
                    opportunities.append(opportunity)
            
            # Get new tokens
            new_tokens = await self.get_new_tokens(6)  # Last 6 hours
            
            for token in new_tokens:
                opportunity = await self._analyze_token_opportunity(token)
                if opportunity:
                    opportunities.append(opportunity)
            
            return sorted(opportunities, key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding trading opportunities: {e}")
            return []
    
    async def _analyze_token_opportunity(self, token: TokenInfo) -> Optional[Dict]:
        """Analyze a token for trading opportunities"""
        try:
            # Get price history
            price_history = await self.get_token_price_history(token.address, '5m', 100)
            
            if len(price_history) < 20:
                return None
            
            # Calculate metrics
            prices = [float(p['price']) for p in price_history]
            volumes = [float(p.get('volume', 0)) for p in price_history]
            
            # Price momentum
            recent_prices = prices[-10:]
            price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Volume analysis
            avg_volume = sum(volumes) / len(volumes)
            recent_volume = sum(volumes[-5:]) / 5
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility
            price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = sum(price_changes) / len(price_changes)
            
            # Calculate opportunity score
            score = 0.0
            
            # Positive momentum
            if price_momentum > 0.05:  # 5% price increase
                score += 0.3
            
            # High volume
            if volume_ratio > 2.0:
                score += 0.2
            
            # Reasonable volatility
            if 0.01 < volatility < 0.5:
                score += 0.2
            
            # Market cap potential
            if token.market_cap < 100000:  # Small cap with potential
                score += 0.2
            
            # Holder growth potential
            if token.holders < 1000:
                score += 0.1
            
            # Risk adjustment
            score -= token.risk_score * 0.3
            
            if score > 0.3:  # Only return high-scoring opportunities
                return {
                    'token_address': token.address,
                    'symbol': token.symbol,
                    'name': token.name,
                    'score': score,
                    'price': token.price,
                    'market_cap': token.market_cap,
                    'volume_24h': token.volume_24h,
                    'holders': token.holders,
                    'risk_score': token.risk_score,
                    'metrics': {
                        'price_momentum': price_momentum,
                        'volume_ratio': volume_ratio,
                        'volatility': volatility
                    },
                    'recommendation': 'buy' if score > 0.5 else 'watch'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing token opportunity: {e}")
            return None
    
    async def execute_trade(self, token_address: str, side: str, amount: float) -> Dict:
        """Execute a trade (placeholder for actual trading)"""
        try:
            # This would integrate with actual Solana trading
            # For now, return a simulated trade
            trade_id = f"trade_{int(time.time())}"
            
            return {
                'trade_id': trade_id,
                'token_address': token_address,
                'side': side,
                'amount': amount,
                'status': 'executed',
                'timestamp': datetime.now(),
                'price': await self._get_current_price(token_address)
            }
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _get_current_price(self, token_address: str) -> float:
        """Get current token price"""
        try:
            token_info = await self.get_token_info(token_address)
            return token_info.price if token_info else 0.0
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    def close(self):
        """Close the client"""
        if self.session:
            asyncio.create_task(self.session.close())