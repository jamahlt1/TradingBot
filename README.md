# Trading Bot Platform

## Overview
A full-featured trading bot platform with a backend dashboard, supporting multiple asset classes (forex, metals, stocks, crypto, indices, futures), connecting to both Solana blockchain and MetaTrader API, with advanced strategy management, risk controls, analytics, and machine learning.

## Features
- **User Authentication:** JWT-based registration, login, and profile management.
- **Strategy Engine:**
  - Supports straddle hedging, trend trading, swing trading, scalping, crypto trend, news trading, sentiment trading, pairs trading, stat arb, ICT concepts.
  - All parameters (risk, RR, trailing, exposure, pairs, assets, ML settings, etc.) are changeable and validated.
  - ML integration for signal generation, optimization, and risk management.
- **Account/Exchange Management:** CRUD for accounts, supports Solana and MetaTrader.
- **Trade Execution & History:** Real trade endpoints, history, and analytics.
- **Analytics:** Real P&L, win rate, drawdown, and performance metrics.
- **Integrations:**
  - Solana wallet (using `solana-py`)
  - CoinGecko price data (using `httpx`)
  - MetaTrader (using `MetaTrader5` or bridge)
  - Bloomberg/Yahoo/RSS news and sentiment
- **Frontend Dashboard:** Modern, responsive UI (React + Material-UI) for all features.

## Backend (FastAPI)
- Python 3.10+
- FastAPI, SQLAlchemy, Alembic, Celery, PostgreSQL, Redis
- Real API endpoints for all features and integrations

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Frontend (React + TypeScript)
- React, Material-UI
- Full dashboard for strategy, account, trade, analytics, and settings management

### Setup
```bash
cd frontend
npm install
npm start
```

## API Usage
- All endpoints are documented via OpenAPI/Swagger at `/docs` when the backend is running.
- Example endpoints:
  - `/auth/register`, `/auth/login`, `/auth/me`
  - `/strategies/`, `/strategies/{id}`, `/strategies/types`, `/strategies/type/{type}`
  - `/accounts/`, `/accounts/{id}`
  - `/trades/`, `/trades/{id}`
  - `/analytics/overview`, `/analytics/performance`, `/analytics/strategy/{id}`
  - `/solana/`, `/coingecko/`, `/metatrader/`, `/news/`

## Contribution Guidelines
- Fork the repo and create a feature branch.
- Write clear, production-quality code with type hints and docstrings.
- Add tests for new features.
- Submit a pull request with a clear description.

---

This is a work in progress. Contributions and feedback are welcome!