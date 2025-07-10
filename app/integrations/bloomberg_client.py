#!/usr/bin/env python3
"""
Bloomberg Client - Alternative Implementation
Uses alternative data sources since Bloomberg Terminal requires special licensing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

logger = logging.getLogger(__name__)

class BloombergClient:
    """
    Bloomberg-style client using alternative data sources
    Provides professional-grade financial data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.alpha_vantage_key = self.config.get('alpha_vantage_api_key')
        self.session = None
        
        # Initialize Alpha Vantage clients
        if self.alpha_vantage_key:
            self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        else:
            self.ts = None
            self.fd = None
            logger.warning("Alpha Vantage API key not provided, functionality will be limited")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_equity_data(self, symbol: str, period: str = '1y') -> Optional[Dict]:
        """Get equity data using Yahoo Finance as primary source"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get historical data
            hist = ticker.history(period=period)
            
            # Get financial statements
            try:
                financials = ticker.financials
                balance_sheet = ticker.balance_sheet
                cashflow = ticker.cashflow
            except:
                financials = balance_sheet = cashflow = None
            
            return {
                'symbol': symbol,
                'info': info,
                'historical_data': hist.to_dict('records') if not hist.empty else [],
                'financials': financials.to_dict() if financials is not None else {},
                'balance_sheet': balance_sheet.to_dict() if balance_sheet is not None else {},
                'cashflow': cashflow.to_dict() if cashflow is not None else {},
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get equity data for {symbol}: {e}")
            return None
    
    async def get_bond_data(self, symbol: str) -> Optional[Dict]:
        """Get bond data"""
        try:
            # Use Yahoo Finance for bond ETFs and treasury data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='1y')
            
            return {
                'symbol': symbol,
                'type': 'bond',
                'info': info,
                'historical_data': hist.to_dict('records') if not hist.empty else [],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get bond data for {symbol}: {e}")
            return None
    
    async def get_fx_data(self, pair: str) -> Optional[Dict]:
        """Get FX data"""
        try:
            # Use Alpha Vantage for FX data if available
            if self.ts:
                try:
                    data, meta_data = self.ts.get_currency_exchange_daily(
                        from_symbol=pair[:3],
                        to_symbol=pair[3:],
                        outputsize='compact'
                    )
                    
                    return {
                        'pair': pair,
                        'type': 'fx',
                        'data': data.to_dict('records'),
                        'meta_data': meta_data,
                        'last_updated': datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Alpha Vantage FX request failed: {e}")
            
            # Fallback to Yahoo Finance
            ticker = yf.Ticker(f"{pair}=X")
            hist = ticker.history(period='1y')
            
            return {
                'pair': pair,
                'type': 'fx',
                'historical_data': hist.to_dict('records') if not hist.empty else [],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get FX data for {pair}: {e}")
            return None
    
    async def get_commodity_data(self, symbol: str) -> Optional[Dict]:
        """Get commodity data"""
        try:
            # Map commodity symbols
            commodity_map = {
                'GOLD': 'GC=F',
                'SILVER': 'SI=F',
                'OIL': 'CL=F',
                'BRENT': 'BZ=F',
                'COPPER': 'HG=F',
                'WHEAT': 'ZW=F',
                'CORN': 'ZC=F'
            }
            
            yahoo_symbol = commodity_map.get(symbol.upper(), symbol)
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period='1y')
            info = ticker.info
            
            return {
                'symbol': symbol,
                'yahoo_symbol': yahoo_symbol,
                'type': 'commodity',
                'info': info,
                'historical_data': hist.to_dict('records') if not hist.empty else [],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get commodity data for {symbol}: {e}")
            return None
    
    async def get_economic_data(self, indicator: str) -> Optional[Dict]:
        """Get economic indicators"""
        try:
            if not self.alpha_vantage_key:
                logger.error("Alpha Vantage API key required for economic data")
                return None
            
            # Map common indicators
            indicator_map = {
                'GDP': 'REAL_GDP',
                'UNEMPLOYMENT': 'UNEMPLOYMENT',
                'CPI': 'CPI',
                'FEDERAL_FUNDS_RATE': 'FEDERAL_FUNDS_RATE',
                'TREASURY_YIELD_10Y': 'TREASURY_YIELD'
            }
            
            av_indicator = indicator_map.get(indicator.upper(), indicator)
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'REAL_GDP',
                'interval': 'annual',
                'apikey': self.alpha_vantage_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'indicator': indicator,
                        'data': data,
                        'last_updated': datetime.now().isoformat()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get economic data for {indicator}: {e}")
            return None
    
    async def get_earnings_data(self, symbol: str) -> Optional[Dict]:
        """Get earnings data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get earnings data
            earnings = ticker.earnings
            quarterly_earnings = ticker.quarterly_earnings
            
            # Get next earnings date
            calendar = ticker.calendar
            
            return {
                'symbol': symbol,
                'annual_earnings': earnings.to_dict() if earnings is not None else {},
                'quarterly_earnings': quarterly_earnings.to_dict() if quarterly_earnings is not None else {},
                'earnings_calendar': calendar.to_dict() if calendar is not None else {},
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get earnings data for {symbol}: {e}")
            return None
    
    async def get_analyst_ratings(self, symbol: str) -> Optional[Dict]:
        """Get analyst ratings and recommendations"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recommendations
            recommendations = ticker.recommendations
            
            return {
                'symbol': symbol,
                'recommendations': recommendations.to_dict() if recommendations is not None else {},
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get analyst ratings for {symbol}: {e}")
            return None
    
    async def get_options_data(self, symbol: str, expiry: Optional[str] = None) -> Optional[Dict]:
        """Get options data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get options expirations
            expirations = ticker.options
            
            if not expirations:
                return None
            
            # Use provided expiry or first available
            target_expiry = expiry or expirations[0]
            
            # Get options chain
            options = ticker.option_chain(target_expiry)
            
            return {
                'symbol': symbol,
                'expiry': target_expiry,
                'available_expirations': list(expirations),
                'calls': options.calls.to_dict('records'),
                'puts': options.puts.to_dict('records'),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get options data for {symbol}: {e}")
            return None
    
    async def get_sector_performance(self) -> Optional[Dict]:
        """Get sector performance data"""
        try:
            # Use sector ETFs as proxies
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Industrials': 'XLI',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Utilities': 'XLU',
                'Communication Services': 'XLC'
            }
            
            sector_data = {}
            
            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period='1y')
                    
                    if not hist.empty:
                        # Calculate performance metrics
                        current_price = hist['Close'].iloc[-1]
                        year_start = hist['Close'].iloc[0]
                        ytd_return = ((current_price - year_start) / year_start) * 100
                        
                        sector_data[sector] = {
                            'etf_symbol': etf,
                            'current_price': current_price,
                            'ytd_return': ytd_return,
                            'volume': hist['Volume'].iloc[-1]
                        }
                except Exception as e:
                    logger.warning(f"Failed to get data for sector {sector}: {e}")
            
            return {
                'sectors': sector_data,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get sector performance: {e}")
            return None
    
    async def get_market_indices(self) -> Optional[Dict]:
        """Get major market indices"""
        try:
            indices = {
                'S&P 500': '^GSPC',
                'Dow Jones': '^DJI',
                'NASDAQ': '^IXIC',
                'Russell 2000': '^RUT',
                'VIX': '^VIX',
                'FTSE 100': '^FTSE',
                'DAX': '^GDAXI',
                'Nikkei': '^N225',
                'Hang Seng': '^HSI'
            }
            
            index_data = {}
            
            for name, symbol in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change = current - previous
                        change_pct = (change / previous) * 100
                        
                        index_data[name] = {
                            'symbol': symbol,
                            'current': current,
                            'change': change,
                            'change_percent': change_pct,
                            'volume': hist['Volume'].iloc[-1] if 'Volume' in hist else 0
                        }
                except Exception as e:
                    logger.warning(f"Failed to get data for index {name}: {e}")
            
            return {
                'indices': index_data,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get market indices: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1d', limit: int = 100) -> List[Dict]:
        """Get historical data in trading format"""
        try:
            # Convert timeframe
            period_map = {
                '1d': '1y', '1w': '2y', '1M': '5y'
            }
            period = period_map.get(timeframe, '1y')
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=timeframe if timeframe != '1w' else '1wk')
            
            if hist.empty:
                return []
            
            # Convert to standard format
            result = []
            for index, row in hist.tail(limit).iterrows():
                result.append({
                    'timestamp': index.isoformat(),
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume']
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    async def get_news(self, symbol: Optional[str] = None, limit: int = 10) -> Optional[List[Dict]]:
        """Get financial news"""
        try:
            if symbol:
                ticker = yf.Ticker(symbol)
                news = ticker.news
            else:
                # Get general market news from a major index
                ticker = yf.Ticker('^GSPC')
                news = ticker.news
            
            # Format news data
            formatted_news = []
            for item in news[:limit]:
                formatted_news.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'publisher': item.get('publisher', ''),
                    'published_date': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'type': item.get('type', '')
                })
            
            return formatted_news
            
        except Exception as e:
            logger.error(f"Failed to get news: {e}")
            return None
    
    def close(self):
        """Close the session"""
        if self.session:
            asyncio.create_task(self.session.close())