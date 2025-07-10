# Trading Strategy Guide

## Overview
This guide covers all strategies available on the platform, their parameters, settings, and how to optimize or customize them. It also explains how to use AI/LLM (DeepSeek/OpenRouter) for suggestions and tuning.

---

## 1. Strategy Categories & List

### Cryptocurrency
- Trend Following
- Mean Reversion
- Breakout
- Arbitrage
- News Trading
- Scalping
- Grid Trading
- Momentum
- Volatility
- Correlation

### Forex
- Trend Following
- Mean Reversion
- Breakout
- Scalping
- News Trading
- Carry Trade
- Momentum
- Range Trading
- Volatility
- Correlation

### Stocks
- Trend Following
- Mean Reversion
- Breakout
- Scalping
- Earnings Trading
- Momentum
- Value Investing
- Growth Investing
- Dividend
- Sector Rotation

### Futures
- Trend Following
- Mean Reversion
- Breakout
- Scalping
- Spread Trading
- Momentum
- Volatility
- Seasonal
- Arbitrage
- Options

### Metals
- Gold Trend Following
- Silver Mean Reversion
- Platinum Breakout
- Palladium Scalping
- Metals Correlation
- Momentum
- Volatility
- News Trading
- Seasonal
- Hedging

### Indices
- S&P 500 Trend Following
- NASDAQ Mean Reversion
- DOW Breakout
- FTSE Scalping
- DAX Momentum
- Nikkei Volatility
- Hang Seng Correlation
- CAC News Trading
- ASX Seasonal
- BSE Hedging

---

## 2. Common Parameters & Settings

- **risk_per_trade**: % of capital risked per trade (e.g. 0.01 = 1%)
- **max_daily_loss**: Max % loss per day before pausing
- **max_drawdown**: Max % drawdown before pausing
- **position_size_method**: fixed, kelly, martingale, risk_based
- **stop_loss_pct**: % stop loss
- **take_profit_pct**: % take profit
- **trailing_stop**: Enable trailing stop (true/false)
- **trailing_stop_pct**: % trailing stop
- **max_positions**: Max open positions
- **correlation_limit**: Max allowed correlation between positions
- **indicators**: List of technical indicators used
- **timeframe**: Chart timeframe (1m, 5m, 1h, 4h, 1d, etc.)
- **symbols**: List of assets/pairs

---

## 3. Example Configurations

### Conservative Trend Following (Forex)
```json
{
  "strategy_name": "Conservative Trend Following",
  "asset_class": "forex",
  "risk_per_trade": 0.01,
  "max_daily_loss": 0.03,
  "max_drawdown": 0.05,
  "position_size_method": "risk_based",
  "stop_loss_pct": 0.015,
  "take_profit_pct": 0.03,
  "trailing_stop": true,
  "trailing_stop_pct": 0.005,
  "indicators": {"ema": [20, 50], "rsi": 14},
  "symbols": ["EURUSD", "GBPUSD"]
}
```

### Aggressive Breakout (Crypto)
```json
{
  "strategy_name": "Aggressive Breakout",
  "asset_class": "crypto",
  "risk_per_trade": 0.05,
  "max_daily_loss": 0.10,
  "max_drawdown": 0.20,
  "position_size_method": "kelly",
  "stop_loss_pct": 0.03,
  "take_profit_pct": 0.06,
  "trailing_stop": true,
  "trailing_stop_pct": 0.01,
  "indicators": {"bollinger_bands": {"period": 20, "std": 2}, "rsi": 14},
  "symbols": ["BTCUSDT", "ETHUSDT"]
}
```

### Compounding with Withdrawals
```json
{
  "strategy_name": "Compounding Grid",
  "asset_class": "crypto",
  "compounding_enabled": true,
  "withdrawal_percentage": 0.1,
  "compounding_rate": 0.05,
  "risk_per_trade": 0.02,
  "max_drawdown": 0.10,
  "position_size_method": "fixed",
  "fixed_amount": 100,
  "symbols": ["BTCUSDT"]
}
```

---

## 4. Uploading Trading Plans
- Go to the Strategies section in the GUI.
- Click “Upload Trading Plan” and select your JSON or CSV file.
- The platform will parse and validate your plan.
- You can then assign it to an account, set allocations, and activate it.

---

## 5. Using DeepSeek/OpenRouter for Suggestions
- In the strategy creator, click “AI Suggest” to get optimal parameters.
- The LLM will analyze your trading plan, recent performance, and market conditions.
- It will recommend:
  - Best strategy type for current market
  - Optimal risk and sizing settings
  - Technical indicator tweaks
  - When to compound or withdraw
- You can accept, edit, or re-run suggestions.
- All suggestions are logged for review.

---

## 6. FAQ & Troubleshooting
- **Q:** My strategy is not performing as expected?
  - **A:** Use the AI “Review” button to get a full analysis and suggestions.
- **Q:** How do I set up balance protection?
  - **A:** Use the Balance Protection panel in the dashboard or CLI.
- **Q:** Can I run multiple strategies at once?
  - **A:** Yes, assign allocation % to each strategy per account.
- **Q:** How do I pause or emergency stop all strategies?
  - **A:** Use the Failsafe panel in the GUI or `cli stop-all`.

---

## 7. Optimization Tips
- Use backtesting to compare strategies before going live.
- Let the AI/LLM review your strategy weekly for parameter tuning.
- Use compounding with periodic withdrawals for steady growth.
- Diversify across asset classes and strategies.
- Monitor correlation and avoid over-concentration.

---

For more details, see the [AI/ML Guide](ai_ml_guide.md) and [API Guide](api_guide.md).