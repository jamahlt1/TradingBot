# 🤖 Enterprise Trading Bot Platform

**Production-Grade, Multi-Strategy Trading Bot with Advanced ML, Risk Management, and Real-Time Analytics**

## 🚀 Features

### **Core Trading Strategies**
- **Trend Following** - Multi-timeframe analysis with ML optimization
- **Swing Trading** - Support/resistance with swing point detection
- **Pairs Trading** - Statistical arbitrage with cointegration
- **Scalping** - High-frequency trading with micro-timing
- **Crypto Arbitrage** - Multi-exchange with correlation analysis
- **News-Based Trading** - Sentiment analysis with event detection
- **ICT Strategy** - Institutional order flow and market structure
- **TWAP** - Time-weighted average price execution
- **Hedging** - Portfolio risk management with dynamic ratios

### **Advanced ML & Analytics**
- **Bayesian Optimization** - Hyperparameter tuning for all strategies
- **DeepSeek LLM Integration** - Real-time market analysis
- **OpenRouter API** - Multi-model AI analysis
- **Technical Indicators** - 50+ indicators with real-time calculation
- **Risk Management** - Position sizing, stop-loss, and portfolio protection

### **Integrations**
- **Solana Blockchain** - Pump.fun token trading
- **MetaTrader API** - Forex and CFD trading
- **CoinGecko** - Cryptocurrency data
- **Bloomberg** - Professional market data
- **Yahoo Finance** - Stock market data
- **RSS Feeds** - News sentiment analysis

### **Risk Management**
- **Emergency Position Closes** - Instant risk mitigation
- **Portfolio Hedging** - Dynamic correlation-based hedging
- **Position Monitoring** - Real-time risk assessment
- **Dollar Cost Averaging** - Systematic investment approach
- **Beta Analysis** - Market correlation tracking

## 📊 Dashboard Features

### **Real-Time Monitoring**
- Live portfolio tracking with P&L
- Strategy performance analytics
- Risk level indicators
- Market sentiment analysis
- Position correlation matrix

### **Advanced Analytics**
- Sharpe ratio and risk metrics
- Drawdown analysis
- Volatility tracking
- Market structure analysis
- Order flow visualization

## 🛠 Installation

### **Prerequisites**
```bash
# Python 3.9+
python --version

# Node.js 16+ (for frontend)
node --version

# PostgreSQL (recommended) or SQLite
```

### **Quick Start**
```bash
# Clone repository
git clone <repository-url>
cd trading-bot-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m alembic upgrade head

# Start the application
python main.py
```

### **Environment Setup**
```bash
# Copy environment template
cp .env.example .env

# Configure your environment variables
OPENROUTER_API_KEY=your_openrouter_key
DEEPSEEK_API_KEY=your_deepseek_key
SOLANA_RPC_URL=your_solana_rpc
METATRADER_CREDENTIALS=your_mt_credentials
```

## 🚀 Usage

### **CLI Interface**
```bash
# Interactive mode
python cli.py

# Dashboard mode
python cli.py --mode dashboard

# Analyze symbol
python cli.py --action analyze --symbol AAPL

# Execute trade
python cli.py --action trade --symbol BTC --side buy --size 0.1

# Optimize strategy
python cli.py --strategy trend_following
```

### **Web Interface**
```bash
# Start web server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access dashboard
http://localhost:8000
```

### **API Endpoints**

#### **Strategy Management**
```bash
# List strategies
GET /api/strategies

# Create strategy
POST /api/strategies
{
  "name": "My Trend Strategy",
  "type": "trend_following",
  "parameters": {
    "risk_per_trade": 0.02,
    "stop_loss_atr": 2.0
  }
}

# Run strategy
POST /api/strategies/{strategy_id}/run

# Backtest strategy
POST /api/strategies/{strategy_id}/backtest
```

#### **Portfolio Management**
```bash
# Get portfolio summary
GET /api/portfolio

# Get positions
GET /api/positions

# Execute trade
POST /api/trades
{
  "symbol": "AAPL",
  "side": "buy",
  "size": 100,
  "strategy_id": "strategy_123"
}
```

#### **Risk Management**
```bash
# Emergency close all positions
POST /api/risk/emergency-close

# Get risk alerts
GET /api/risk/alerts

# Update hedging
POST /api/risk/hedge
```

## 📈 Strategy Details

### **Trend Following Strategy**
- **Multi-timeframe analysis** (1m, 5m, 15m, 1h, 4h, 1d)
- **Dynamic trend detection** using linear regression
- **Risk-adjusted position sizing** based on ATR
- **ML-optimized parameters** with Bayesian optimization

### **Crypto Arbitrage Strategy**
- **Multi-exchange monitoring** (Binance, Coinbase, Kraken)
- **Correlation analysis** to identify arbitrage opportunities
- **Slippage protection** and execution optimization
- **Real-time opportunity detection**

### **News-Based Trading**
- **Real-time sentiment analysis** using LLM
- **Event detection** and impact assessment
- **Market reaction prediction**
- **Risk-adjusted position sizing**

### **ICT Strategy**
- **Market structure analysis** (higher highs, lower lows)
- **Order block identification** for institutional zones
- **Fair value gap detection**
- **Liquidity zone analysis**

### **TWAP Strategy**
- **Large order execution** with minimal market impact
- **Volume profile analysis** for optimal timing
- **Dynamic slice sizing** based on market conditions
- **Real-time execution monitoring**

### **Hedging Strategy**
- **Portfolio beta calculation**
- **Dynamic correlation analysis**
- **Risk-adjusted hedge ratios**
- **Automatic rebalancing**

## 🔧 Configuration

### **Strategy Parameters**
```python
# Example strategy configuration
strategy_config = {
    "trend_following": {
        "short_window": 20,
        "long_window": 50,
        "rsi_period": 14,
        "risk_per_trade": 0.02,
        "max_position_size": 0.1
    },
    "crypto_arbitrage": {
        "min_spread": 0.005,
        "correlation_threshold": 0.7,
        "max_slippage": 0.002
    }
}
```

### **Risk Management Settings**
```python
risk_config = {
    "max_portfolio_risk": 0.05,
    "emergency_stop_loss": 0.10,
    "max_drawdown": 0.20,
    "position_size_limit": 0.15
}
```

## 📊 Performance Metrics

### **Key Performance Indicators**
- **Sharpe Ratio** - Risk-adjusted returns
- **Maximum Drawdown** - Worst peak-to-trough decline
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / Gross loss
- **Calmar Ratio** - Annual return / Maximum drawdown

### **Risk Metrics**
- **Value at Risk (VaR)** - Potential loss at confidence level
- **Expected Shortfall** - Average loss beyond VaR
- **Beta** - Market correlation
- **Volatility** - Price fluctuation measure

## 🔒 Security Features

### **Authentication & Authorization**
- **JWT-based authentication**
- **Role-based access control**
- **API key management**
- **Session management**

### **Data Protection**
- **Encrypted database connections**
- **Secure API key storage**
- **Audit logging**
- **Data backup and recovery**

## 🚨 Emergency Controls

### **Risk Management**
```bash
# Emergency close all positions
python cli.py --emergency-close

# Pause all strategies
python cli.py --pause-all

# Resume strategies
python cli.py --resume-all

# Get risk status
python cli.py --risk-status
```

### **Position Management**
- **Instant emergency closes**
- **Automatic hedging**
- **Risk level monitoring**
- **Portfolio protection**

## 📱 Frontend Features

### **Real-Time Dashboard**
- **Live portfolio tracking**
- **Strategy performance charts**
- **Risk level indicators**
- **Market sentiment display**

### **Strategy Management**
- **Strategy creation and editing**
- **Parameter optimization**
- **Performance backtesting**
- **Real-time monitoring**

### **Advanced Analytics**
- **Correlation matrix visualization**
- **Risk heat maps**
- **Performance attribution**
- **Market structure analysis**

## 🔄 API Integrations

### **Data Sources**
- **Yahoo Finance** - Stock market data
- **CoinGecko** - Cryptocurrency prices
- **Bloomberg** - Professional market data
- **RSS Feeds** - News sentiment

### **Trading Platforms**
- **MetaTrader** - Forex and CFD trading
- **Solana** - Blockchain trading
- **Multiple exchanges** - Crypto arbitrage

### **AI Services**
- **OpenRouter** - Multi-model AI analysis
- **DeepSeek** - Advanced market analysis
- **Custom ML models** - Strategy optimization

## 📚 Documentation

### **API Documentation**
- **Swagger UI** at `/docs`
- **ReDoc** at `/redoc`
- **OpenAPI specification**

### **Strategy Documentation**
- **Parameter guides** for each strategy
- **Backtesting results** and optimization
- **Risk management** procedures

### **Deployment Guide**
- **Docker deployment**
- **Cloud hosting** (AWS, GCP, Azure)
- **Scaling strategies**

## 🤝 Contributing

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black app/
isort app/

# Type checking
mypy app/
```

### **Code Standards**
- **Type hints** for all functions
- **Comprehensive docstrings**
- **Unit tests** for all modules
- **Integration tests** for APIs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### **Getting Help**
- **Documentation** - Comprehensive guides and examples
- **Issues** - Bug reports and feature requests
- **Discussions** - Community support and questions

### **Contact**
- **Email** - support@tradingbot.com
- **Discord** - Community server
- **GitHub** - Issues and discussions

---

## 🎯 Quick Start Example

```python
from app.strategies.engine import TrendFollowingStrategy
from app.risk_management.position_manager import PositionManager

# Initialize strategy
strategy = TrendFollowingStrategy(
    risk_per_trade=0.02,
    stop_loss_atr=2.0,
    take_profit_atr=4.0
)

# Get market data
data = get_market_data("AAPL")

# Generate signals
signals = strategy.generate_signals(data)

# Execute trades
for signal in signals:
    execute_trade(signal)
```

**Ready to start trading?** 🚀

Check out the [Quick Start Guide](docs/quickstart.md) for detailed instructions!

---

## 🚀 Docker Deployment

### Prerequisites
- Docker and Docker Compose installed

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <project-root>
   ```

2. **Build and start all services:**
   ```bash
   docker-compose up --build -d
   ```
   This will start:
   - Backend API (FastAPI, Uvicorn) on [http://localhost:8000](http://localhost:8000)
   - Frontend (React/Bootstrap) on [http://localhost:8080](http://localhost:8080)
   - PostgreSQL database on port 5432

3. **Initialize the database (if needed):**
   - Run migrations or setup scripts as required by your backend (see docs or scripts/setup_database.py if present).

4. **Access the platform:**
   - **Frontend:** [http://localhost:8080](http://localhost:8080)
   - **Backend API:** [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

5. **Stop all services:**
   ```bash
   docker-compose down
   ```

---

## 🧮 CAPM Analytics (Capital Asset Pricing Model)

The platform now supports full CAPM analytics for both individual assets and user portfolios.

### API Endpoints
- **Asset CAPM:**
  - `GET /analytics/capm/asset?symbol=BTCUSDT&market_symbol=SPY&risk_free_rate=0.03`
- **Portfolio CAPM:**
  - `GET /analytics/capm/portfolio?market_symbol=SPY&risk_free_rate=0.03`

**Parameters:**
- `symbol`: Asset symbol (e.g. BTCUSDT, AAPL)
- `market_symbol`: Market index symbol (e.g. SPY, BTCUSD)
- `risk_free_rate`: Annual risk-free rate (e.g. 0.03 for 3%)
- `lookback_days`: Number of days for historical returns (default: 252)

**Returns:**
- `beta`: Asset/portfolio beta
- `expected_return`: CAPM expected return
- `realized_return`: Actual annualized return
- `alpha`: Alpha (excess return)
- `risk_premium`: Market risk premium

### Usage
- Use the API endpoints above to analyze risk/return for any asset or your portfolio.
- CAPM analytics are available in the dashboard, strategy, and portfolio analytics screens.

---

## 🛠️ Troubleshooting
- If you encounter issues, check logs with:
  ```bash
  docker-compose logs backend
  docker-compose logs frontend
  docker-compose logs db
  ```
- Ensure ports 8000, 8080, and 5432 are free.
- For database errors, ensure migrations are applied and the database is healthy.

---

For more, see the [API Guide](docs/api_guide.md), [CLI Guide](docs/cli_guide.md), and [GUI Guide](docs/gui_guide.md).