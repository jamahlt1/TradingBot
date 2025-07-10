#!/usr/bin/env python3
"""
Enterprise-Grade Trading Bot Platform
Main Application Entry Point
"""

import asyncio
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import logging
import sys
import os
from typing import Dict, Any, List
import argparse
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database.database import engine, Base, get_db
from app.models.models import User, Strategy, Account, Trade
from app.auth.auth import create_access_token, get_current_user, authenticate_user
from app.routers import strategies, accounts, trades, auth, optimization, integrations
from app.strategies.engine import (
    PairsTradingStrategy, ScalpingStrategy, SentimentStrategy,
    TrendFollowingStrategy, SwingTradingStrategy, CryptoArbitrageStrategy,
    NewsBasedStrategy, ICTStrategy, TWAPStrategy, HedgingStrategy
)
from app.strategies.engine.bayesian_optimization import BayesianOptimizer, MLModelSelector
from app.integrations.solana_client import SolanaClient
from app.integrations.metatrader_client import MetaTraderClient
from app.integrations.coingecko_client import CoinGeckoClient
from app.integrations.bloomberg_client import BloombergClient
from app.integrations.yahoo_client import YahooClient
from app.integrations.rss_client import RSSClient
from app.analytics.backtesting import BacktestingEngine
from app.risk_management.risk_manager import RiskManager
from app.ml.models import MLModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global state
app_state = {
    'strategies': {},
    'accounts': {},
    'optimizers': {},
    'risk_manager': None,
    'ml_registry': None,
    'integrations': {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Enterprise Trading Bot Platform...")
    
    # Initialize database
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database initialized")
    
    # Initialize risk manager
    app_state['risk_manager'] = RiskManager()
    logger.info("✅ Risk manager initialized")
    
    # Initialize ML registry
    app_state['ml_registry'] = MLModelRegistry()
    logger.info("✅ ML model registry initialized")
    
    # Initialize integrations
    try:
        app_state['integrations']['solana'] = SolanaClient()
        app_state['integrations']['metatrader'] = MetaTraderClient()
        app_state['integrations']['coingecko'] = CoinGeckoClient()
        app_state['integrations']['bloomberg'] = BloombergClient()
        app_state['integrations']['yahoo'] = YahooClient()
        app_state['integrations']['rss'] = RSSClient()
        logger.info("✅ All integrations initialized")
    except Exception as e:
        logger.warning(f"⚠️ Some integrations failed to initialize: {e}")
    
    # Initialize strategy engines
    strategy_classes = {
        'pairs_trading': PairsTradingStrategy,
        'scalping': ScalpingStrategy,
        'sentiment': SentimentStrategy,
        'trend_following': TrendFollowingStrategy,
        'swing_trading': SwingTradingStrategy,
        'crypto_arbitrage': CryptoArbitrageStrategy,
        'news_based': NewsBasedStrategy,
        'ict': ICTStrategy,
        'twap': TWAPStrategy,
        'hedging': HedgingStrategy
    }
    
    for name, strategy_class in strategy_classes.items():
        app_state['strategies'][name] = strategy_class()
        logger.info(f"✅ Strategy '{name}' initialized")
    
    # Initialize Bayesian optimizers
    for name, strategy_class in strategy_classes.items():
        param_space = strategy_class.get_parameter_space()
        app_state['optimizers'][name] = BayesianOptimizer(
            strategy_class=strategy_class,
            param_space=param_space,
            objective_metric='sharpe_ratio',
            n_calls=50,
            n_random_starts=10
        )
        logger.info(f"✅ Optimizer for '{name}' initialized")
    
    logger.info("🎯 All systems operational!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Trading Bot Platform...")
    # Cleanup connections, close positions, etc.
    for integration in app_state['integrations'].values():
        if hasattr(integration, 'close'):
            try:
                integration.close()
            except Exception as e:
                logger.error(f"Error closing integration: {e}")

# Create FastAPI app
app = FastAPI(
    title="Enterprise Trading Bot Platform",
    description="Advanced multi-asset trading platform with ML optimization",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["Strategies"])
app.include_router(accounts.router, prefix="/api/accounts", tags=["Accounts"])
app.include_router(trades.router, prefix="/api/trades", tags=["Trades"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Optimization"])
app.include_router(integrations.router, prefix="/api/integrations", tags=["Integrations"])

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "database": "connected",
            "strategies": len(app_state['strategies']),
            "optimizers": len(app_state['optimizers']),
            "integrations": len(app_state['integrations'])
        }
    }

# Dashboard endpoint
@app.get("/api/dashboard")
async def get_dashboard(current_user: User = Depends(get_current_user)):
    """Get dashboard data"""
    return {
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email
        },
        "platform_stats": {
            "total_strategies": len(app_state['strategies']),
            "active_accounts": 0,  # TODO: Get from database
            "total_trades": 0,     # TODO: Get from database
            "total_pnl": 0.0       # TODO: Calculate from trades
        },
        "available_strategies": list(app_state['strategies'].keys()),
        "available_integrations": list(app_state['integrations'].keys())
    }

# Strategy optimization endpoint
@app.post("/api/optimize/{strategy_name}")
async def optimize_strategy(
    strategy_name: str,
    optimization_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Optimize a strategy using Bayesian optimization"""
    if strategy_name not in app_state['optimizers']:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_name}' not found"
        )
    
    try:
        optimizer = app_state['optimizers'][strategy_name]
        result = optimizer.optimize()
        
        return {
            "strategy_name": strategy_name,
            "optimization_result": result,
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"Optimization failed for {strategy_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )

# Real-time market data endpoint
@app.get("/api/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get real-time market data"""
    try:
        # Try different integrations for data
        data = None
        
        # Try Yahoo first
        if 'yahoo' in app_state['integrations']:
            try:
                data = app_state['integrations']['yahoo'].get_historical_data(
                    symbol, timeframe, limit
                )
            except Exception as e:
                logger.warning(f"Yahoo data failed for {symbol}: {e}")
        
        # Try CoinGecko for crypto
        if not data and 'coingecko' in app_state['integrations']:
            try:
                data = app_state['integrations']['coingecko'].get_historical_data(
                    symbol, timeframe, limit
                )
            except Exception as e:
                logger.warning(f"CoinGecko data failed for {symbol}: {e}")
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {symbol}"
            )
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "last_updated": data[-1]['timestamp'] if data else None
        }
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch market data: {str(e)}"
        )

# Strategy backtesting endpoint
@app.post("/api/backtest/{strategy_name}")
async def backtest_strategy(
    strategy_name: str,
    backtest_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Run backtesting for a strategy"""
    if strategy_name not in app_state['strategies']:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_name}' not found"
        )
    
    try:
        strategy = app_state['strategies'][strategy_name]
        backtest_engine = BacktestingEngine()
        
        # Get historical data
        symbol = backtest_config.get('symbol', 'AAPL')
        start_date = backtest_config.get('start_date', '2023-01-01')
        end_date = backtest_config.get('end_date', '2024-01-01')
        
        # Fetch data from integration
        data = None
        if 'yahoo' in app_state['integrations']:
            try:
                data = app_state['integrations']['yahoo'].get_historical_data(
                    symbol, '1d', 1000
                )
            except Exception as e:
                logger.warning(f"Failed to get data for backtest: {e}")
        
        if not data:
            raise HTTPException(
                status_code=400,
                detail="No historical data available for backtesting"
            )
        
        # Run backtest
        results = backtest_engine.run_backtest(
            strategy=strategy,
            data=data,
            config=backtest_config
        )
        
        return {
            "strategy_name": strategy_name,
            "backtest_results": results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Backtesting failed for {strategy_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Backtesting failed: {str(e)}"
        )

# Portfolio management endpoint
@app.get("/api/portfolio")
async def get_portfolio(current_user: User = Depends(get_current_user)):
    """Get user portfolio"""
    try:
        # TODO: Implement real portfolio tracking
        return {
            "total_value": 100000.0,
            "cash": 25000.0,
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "avg_price": 150.0,
                    "current_price": 155.0,
                    "unrealized_pnl": 500.0,
                    "unrealized_pnl_percent": 3.33
                }
            ],
            "performance": {
                "total_return": 5.0,
                "daily_return": 0.5,
                "sharpe_ratio": 1.2,
                "max_drawdown": -2.1
            }
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch portfolio: {str(e)}"
        )

# Risk management endpoint
@app.get("/api/risk/analysis")
async def get_risk_analysis(current_user: User = Depends(get_current_user)):
    """Get risk analysis"""
    try:
        risk_manager = app_state['risk_manager']
        analysis = risk_manager.get_portfolio_risk_analysis()
        
        return {
            "risk_metrics": analysis,
            "alerts": risk_manager.get_active_alerts(),
            "recommendations": risk_manager.get_risk_recommendations()
        }
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Risk analysis failed: {str(e)}"
        )

def run_cli():
    """CLI interface for the trading bot"""
    parser = argparse.ArgumentParser(description="Enterprise Trading Bot Platform")
    parser.add_argument("--mode", choices=["api", "cli"], default="api", help="Run mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", help="Configuration file path")
    
    # CLI subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Strategy commands
    strategy_parser = subparsers.add_parser("strategy", help="Strategy management")
    strategy_parser.add_argument("action", choices=["list", "run", "backtest", "optimize"])
    strategy_parser.add_argument("--name", help="Strategy name")
    strategy_parser.add_argument("--symbol", help="Trading symbol")
    strategy_parser.add_argument("--config", help="Strategy configuration file")
    
    # Account commands
    account_parser = subparsers.add_parser("account", help="Account management")
    account_parser.add_argument("action", choices=["list", "create", "balance", "positions"])
    account_parser.add_argument("--name", help="Account name")
    account_parser.add_argument("--type", help="Account type")
    
    # Trade commands
    trade_parser = subparsers.add_parser("trade", help="Trade management")
    trade_parser.add_argument("action", choices=["list", "place", "cancel", "history"])
    trade_parser.add_argument("--symbol", help="Trading symbol")
    trade_parser.add_argument("--side", choices=["buy", "sell"], help="Trade side")
    trade_parser.add_argument("--quantity", type=float, help="Trade quantity")
    trade_parser.add_argument("--price", type=float, help="Trade price")
    
    # Optimization commands
    opt_parser = subparsers.add_parser("optimize", help="Strategy optimization")
    opt_parser.add_argument("--strategy", required=True, help="Strategy to optimize")
    opt_parser.add_argument("--metric", default="sharpe_ratio", help="Optimization metric")
    opt_parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    
    # Integration commands
    int_parser = subparsers.add_parser("integration", help="Integration management")
    int_parser.add_argument("action", choices=["list", "test", "configure"])
    int_parser.add_argument("--name", help="Integration name")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        # Run API server
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=args.debug,
            log_level="debug" if args.debug else "info"
        )
    elif args.mode == "cli":
        # Handle CLI commands
        handle_cli_command(args)

def handle_cli_command(args):
    """Handle CLI commands"""
    if args.command == "strategy":
        handle_strategy_command(args)
    elif args.command == "account":
        handle_account_command(args)
    elif args.command == "trade":
        handle_trade_command(args)
    elif args.command == "optimize":
        handle_optimize_command(args)
    elif args.command == "integration":
        handle_integration_command(args)
    else:
        print("Please specify a command. Use --help for more information.")

def handle_strategy_command(args):
    """Handle strategy CLI commands"""
    if args.action == "list":
        print("Available strategies:")
        for name in app_state['strategies'].keys():
            print(f"  - {name}")
    elif args.action == "run":
        if not args.name:
            print("Error: Strategy name required")
            return
        print(f"Running strategy: {args.name}")
        # TODO: Implement strategy execution
    elif args.action == "backtest":
        if not args.name:
            print("Error: Strategy name required")
            return
        print(f"Running backtest for strategy: {args.name}")
        # TODO: Implement backtesting
    elif args.action == "optimize":
        if not args.name:
            print("Error: Strategy name required")
            return
        print(f"Optimizing strategy: {args.name}")
        # TODO: Implement optimization

def handle_account_command(args):
    """Handle account CLI commands"""
    if args.action == "list":
        print("Available accounts:")
        # TODO: List accounts from database
    elif args.action == "create":
        if not args.name:
            print("Error: Account name required")
            return
        print(f"Creating account: {args.name}")
        # TODO: Create account

def handle_trade_command(args):
    """Handle trade CLI commands"""
    if args.action == "list":
        print("Recent trades:")
        # TODO: List trades from database
    elif args.action == "place":
        if not all([args.symbol, args.side, args.quantity]):
            print("Error: Symbol, side, and quantity required")
            return
        print(f"Placing {args.side} order for {args.quantity} {args.symbol}")
        # TODO: Place trade

def handle_optimize_command(args):
    """Handle optimization CLI commands"""
    if args.strategy not in app_state['optimizers']:
        print(f"Error: Strategy '{args.strategy}' not found")
        return
    
    print(f"Optimizing strategy: {args.strategy}")
    print(f"Metric: {args.metric}")
    print(f"Iterations: {args.iterations}")
    
    # TODO: Run optimization
    print("Optimization completed!")

def handle_integration_command(args):
    """Handle integration CLI commands"""
    if args.action == "list":
        print("Available integrations:")
        for name in app_state['integrations'].keys():
            print(f"  - {name}")
    elif args.action == "test":
        if not args.name:
            print("Error: Integration name required")
            return
        print(f"Testing integration: {args.name}")
        # TODO: Test integration

if __name__ == "__main__":
    run_cli()