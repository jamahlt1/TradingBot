# Strategies Guide

## Overview
This guide details all available trading strategies, every setting and parameter, all machine learning models, and every risk management option on the platform. It also provides suggestions for optimal use and configuration.

---

## 1. Strategy Types

### Trend Following
- **Description:** Follows market trends using moving averages, MACD, ADX, etc.
- **Best for:** Trending markets (forex, crypto, stocks)

### Mean Reversion
- **Description:** Buys/sells when price deviates from mean (Bollinger Bands, RSI, etc.)
- **Best for:** Range-bound or mean-reverting assets

### Breakout
- **Description:** Enters trades on price breakouts from support/resistance or volatility bands.
- **Best for:** Volatile markets, news events

### Arbitrage
- **Description:** Exploits price differences across exchanges or pairs.
- **Best for:** Crypto, futures, cross-asset

### Scalping
- **Description:** High-frequency, small profit trades, often using order book or micro-patterns.
- **Best for:** High-liquidity markets

### Swing Trading
- **Description:** Captures multi-day moves using technical and fundamental analysis.
- **Best for:** Stocks, forex, metals

### News-Based
- **Description:** Trades based on news, economic releases, or sentiment.
- **Best for:** All asset classes

### Grid Trading
- **Description:** Places buy/sell orders at regular intervals above/below a set price.
- **Best for:** Sideways or volatile markets

### Hedging
- **Description:** Reduces risk by holding offsetting positions.
- **Best for:** Large portfolios, prop firm challenges

### Pairs Trading
- **Description:** Trades correlated assets for mean reversion.
- **Best for:** Stocks, crypto, forex

### Sentiment Trading
- **Description:** Uses news, social, and AI sentiment to drive trades.
- **Best for:** All asset classes

### High-Frequency Trading (HFT)
- **Description:** Ultra-fast, algorithmic trading for arbitrage, market making, etc.
- **Best for:** Advanced users, low-latency markets

---

## 2. All Settings & Parameters

### General
- **strategy_name**: Name of the strategy
- **asset_class**: crypto, forex, stocks, futures, metals, indices
- **symbols**: List of assets/pairs
- **timeframe**: 1m, 5m, 15m, 1h, 4h, 1d, 1w
- **enabled**: true/false

### Risk Management
- **risk_per_trade**: % of capital risked per trade (e.g. 0.01 = 1%)
- **max_daily_loss**: Max % loss per day before pausing
- **max_drawdown**: Max % drawdown before pausing
- **max_position_size**: Max % of capital in a single position
- **correlation_limit**: Max allowed correlation between open positions
- **stop_loss_pct**: % stop loss
- **take_profit_pct**: % take profit
- **trailing_stop**: Enable trailing stop (true/false)
- **trailing_stop_pct**: % trailing stop
- **max_positions**: Max open positions
- **auto_hedge**: Enable automatic hedging (true/false)
- **emergency_stop**: Enable emergency stop triggers (true/false)

### Position Sizing
- **position_size_method**: fixed, kelly, martingale, risk_based
- **fixed_amount**: Fixed dollar amount per trade
- **kelly_fraction**: Kelly criterion fraction (0-1)
- **martingale_multiplier**: Multiplier for martingale sizing

### Technical Indicators
- **moving_averages**: SMA, EMA, WMA, HMA (periods)
- **oscillators**: RSI, Stochastic, CCI, Williams %R (periods, thresholds)
- **volatility**: Bollinger Bands, ATR, Keltner Channels (periods, std)
- **momentum**: MACD, ADX, ROC (parameters)
- **volume**: VWAP, OBV, volume thresholds

### Advanced
- **compounding_enabled**: true/false
- **compounding_rate**: % of profits to reinvest
- **withdrawal_percentage**: % of profits to withdraw
- **auto_withdrawal**: true/false
- **news_sentiment_enabled**: true/false
- **social_sentiment_enabled**: true/false
- **ai_optimization_enabled**: true/false
- **backtest_before_live**: true/false
- **llm_assist_enabled**: true/false

---

## 3. Machine Learning Models Available

### Random Forest
- **Purpose:** Classification/regression for market direction
- **Parameters:** n_estimators, max_depth, min_samples_split, etc.

### XGBoost
- **Purpose:** Gradient boosting for price prediction
- **Parameters:** n_estimators, max_depth, learning_rate, etc.

### SVM (Support Vector Machine)
- **Purpose:** Classification for buy/sell signals
- **Parameters:** C, kernel, gamma

### Neural Network (MLP)
- **Purpose:** Nonlinear pattern recognition
- **Parameters:** hidden_layer_sizes, activation, solver

### LSTM
- **Purpose:** Time series prediction
- **Parameters:** units, layers, dropout

### Transformer
- **Purpose:** Attention-based sequence modeling
- **Parameters:** d_model, n_heads, n_layers

### CNN
- **Purpose:** Pattern recognition in price/indicator data
- **Parameters:** layers, filters, kernel_size

### Autoencoder
- **Purpose:** Anomaly detection, feature reduction
- **Parameters:** encoder/decoder layers

### GAN
- **Purpose:** Synthetic data generation
- **Parameters:** generator/discriminator layers

---

## 4. Risk Management Settings
- **Daily Loss Limit:** Pauses all trading if daily loss exceeds set %
- **Max Drawdown:** Pauses all trading if drawdown exceeds set %
- **Position Size Limit:** Prevents overexposure to a single asset
- **Correlation Filter:** Avoids highly correlated positions
- **Stop Loss/Take Profit:** Automatic exit at set %
- **Trailing Stop:** Locks in profits as price moves
- **Auto Hedge:** Opens offsetting positions to reduce risk
- **Emergency Stop:** Closes all positions and pauses all strategies
- **Compounding/Withdrawal:** Controls profit reinvestment and withdrawals
- **Backtest Before Live:** Ensures strategy is tested before live trading

---

## 5. Suggestions & Best Practices
- **Use AI/LLM optimization** for parameter tuning and strategy selection.
- **Backtest** every strategy before going live.
- **Diversify** across asset classes and strategies.
- **Enable compounding** for long-term growth, but set withdrawal % for safety.
- **Monitor correlation** to avoid over-concentration.
- **Set conservative risk limits** for prop firm challenges.
- **Enable news/social sentiment** for adaptive strategies.
- **Use emergency stop and failsafe features** for capital protection.

---

For more, see the [AI Guide](Ai.md), [API Guide](APIs.md), and [CLI Guide](CLIs.md).