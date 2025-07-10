#!/usr/bin/env python3
"""
CoinGecko API Client
Provides cryptocurrency market data and pricing information
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class CoinGeckoClient:
    """
    CoinGecko API client for cryptocurrency data
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_url = "https://pro-api.coingecko.com/api/v3"
        self.session = None
        self.rate_limit_delay = 1.0  # seconds between requests for free tier
        self.last_request_time = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make rate-limited API request"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)
            
            # Choose URL based on API key
            base_url = self.pro_url if self.api_key else self.base_url
            url = f"{base_url}/{endpoint}"
            
            headers = {}
            if self.api_key:
                headers['x-cg-pro-api-key'] = self.api_key
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, headers=headers, params=params) as response:
                self.last_request_time = time.time()
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limited, wait and retry
                    logger.warning("Rate limited, waiting...")
                    await asyncio.sleep(60)
                    return await self._make_request(endpoint, params)
                else:
                    logger.error(f"API request failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    async def get_price(self, coin_ids: List[str], vs_currencies: List[str] = ['usd']) -> Optional[Dict]:
        """Get current prices for coins"""
        try:
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': ','.join(vs_currencies),
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            return await self._make_request('simple/price', params)
            
        except Exception as e:
            logger.error(f"Failed to get prices: {e}")
            return None
    
    async def get_coin_data(self, coin_id: str) -> Optional[Dict]:
        """Get detailed coin data"""
        try:
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false',
                'sparkline': 'true'
            }
            
            return await self._make_request(f'coins/{coin_id}', params)
            
        except Exception as e:
            logger.error(f"Failed to get coin data: {e}")
            return None
    
    async def get_market_data(self, coin_id: str, vs_currency: str = 'usd', 
                             days: int = 30, interval: str = 'daily') -> Optional[List[Dict]]:
        """Get historical market data"""
        try:
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': interval
            }
            
            data = await self._make_request(f'coins/{coin_id}/market_chart', params)
            
            if data:
                # Convert to standard format
                prices = data.get('prices', [])
                volumes = data.get('total_volumes', [])
                market_caps = data.get('market_caps', [])
                
                result = []
                for i, price_point in enumerate(prices):
                    timestamp = datetime.fromtimestamp(price_point[0] / 1000)
                    volume = volumes[i][1] if i < len(volumes) else 0
                    market_cap = market_caps[i][1] if i < len(market_caps) else 0
                    
                    result.append({
                        'timestamp': timestamp.isoformat(),
                        'price': price_point[1],
                        'volume': volume,
                        'market_cap': market_cap
                    })
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1d', 
                                 limit: int = 100) -> List[Dict]:
        """Get historical data in trading format"""
        try:
            # Convert symbol to coin ID
            coin_id = await self._symbol_to_coin_id(symbol.lower())
            if not coin_id:
                logger.error(f"Could not find coin ID for symbol: {symbol}")
                return []
            
            # Convert timeframe to days
            timeframe_to_days = {
                '1h': 1, '4h': 7, '1d': min(limit, 365),
                '1w': min(limit * 7, 365), '1M': min(limit * 30, 365)
            }
            days = timeframe_to_days.get(timeframe, 30)
            
            # Get OHLC data if available (premium feature)
            if self.api_key and timeframe in ['1h', '4h', '1d']:
                ohlc_data = await self._make_request(f'coins/{coin_id}/ohlc', {
                    'vs_currency': 'usd',
                    'days': days
                })
                
                if ohlc_data:
                    result = []
                    for candle in ohlc_data[-limit:]:
                        result.append({
                            'timestamp': datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                            'open': candle[1],
                            'high': candle[2],
                            'low': candle[3],
                            'close': candle[4],
                            'volume': 0  # OHLC doesn't include volume
                        })
                    return result
            
            # Fallback to market chart data
            market_data = await self.get_market_data(coin_id, days=days)
            if market_data:
                # Convert price data to OHLC format (simplified)
                result = []
                for i, data_point in enumerate(market_data[-limit:]):
                    result.append({
                        'timestamp': data_point['timestamp'],
                        'open': data_point['price'],
                        'high': data_point['price'],
                        'low': data_point['price'],
                        'close': data_point['price'],
                        'volume': data_point['volume']
                    })
                return result
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    async def _symbol_to_coin_id(self, symbol: str) -> Optional[str]:
        """Convert trading symbol to CoinGecko coin ID"""
        try:
            # Common mappings
            symbol_mapping = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'bnb': 'binancecoin',
                'ada': 'cardano',
                'sol': 'solana',
                'dot': 'polkadot',
                'matic': 'matic-network',
                'avax': 'avalanche-2',
                'link': 'chainlink',
                'uni': 'uniswap'
            }
            
            if symbol in symbol_mapping:
                return symbol_mapping[symbol]
            
            # Search via API
            coins_list = await self._make_request('coins/list')
            if coins_list:
                for coin in coins_list:
                    if coin['symbol'].lower() == symbol.lower():
                        return coin['id']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to map symbol to coin ID: {e}")
            return None
    
    async def get_trending_coins(self) -> Optional[List[Dict]]:
        """Get trending coins"""
        try:
            data = await self._make_request('search/trending')
            if data and 'coins' in data:
                return [coin['item'] for coin in data['coins']]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get trending coins: {e}")
            return None
    
    async def get_top_coins(self, limit: int = 100, vs_currency: str = 'usd') -> Optional[List[Dict]]:
        """Get top coins by market cap"""
        try:
            params = {
                'vs_currency': vs_currency,
                'order': 'market_cap_desc',
                'per_page': min(limit, 250),
                'page': 1,
                'sparkline': 'false',
                'price_change_percentage': '24h'
            }
            
            return await self._make_request('coins/markets', params)
            
        except Exception as e:
            logger.error(f"Failed to get top coins: {e}")
            return None
    
    async def get_exchanges(self) -> Optional[List[Dict]]:
        """Get list of exchanges"""
        try:
            return await self._make_request('exchanges')
            
        except Exception as e:
            logger.error(f"Failed to get exchanges: {e}")
            return None
    
    async def get_exchange_tickers(self, exchange_id: str, coin_ids: Optional[List[str]] = None) -> Optional[Dict]:
        """Get tickers for specific exchange"""
        try:
            params = {}
            if coin_ids:
                params['coin_ids'] = ','.join(coin_ids)
            
            return await self._make_request(f'exchanges/{exchange_id}/tickers', params)
            
        except Exception as e:
            logger.error(f"Failed to get exchange tickers: {e}")
            return None
    
    async def get_global_data(self) -> Optional[Dict]:
        """Get global cryptocurrency data"""
        try:
            return await self._make_request('global')
            
        except Exception as e:
            logger.error(f"Failed to get global data: {e}")
            return None
    
    async def get_fear_greed_index(self) -> Optional[Dict]:
        """Get fear and greed index"""
        try:
            # This uses an alternative API as CoinGecko doesn't provide fear/greed
            url = "https://api.alternative.me/fng/"
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Failed to get fear/greed index: {e}")
            return None
    
    async def search_coins(self, query: str) -> Optional[List[Dict]]:
        """Search for coins"""
        try:
            params = {'query': query}
            data = await self._make_request('search', params)
            
            if data and 'coins' in data:
                return data['coins']
            return None
            
        except Exception as e:
            logger.error(f"Failed to search coins: {e}")
            return None
    
    def close(self):
        """Close the session"""
        if self.session:
            asyncio.create_task(self.session.close())