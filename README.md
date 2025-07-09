# Trading Bot Platform

## Overview
A full-featured trading bot platform with a backend dashboard, supporting multiple asset classes (forex, metals, stocks, crypto, indices, futures), connecting to both Solana blockchain and MetaTrader API, with advanced strategy management, risk controls, analytics, and machine learning.

## Backend (FastAPI)
- Python 3.10+
- FastAPI, SQLAlchemy, Celery, PostgreSQL, Redis

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn Main:app --reload
```

## Frontend (React + TypeScript)
- React, Material-UI

### Setup
```bash
cd frontend
npm install
npm start
```

## Features (Planned)
- User authentication
- Strategy management (CRUD)
- Asset/account management
- Trade execution endpoints
- Risk management settings
- WebSocket for live updates
- Modular strategy/connector plugins
- Dashboard UI with charts and analytics

---

This is a work in progress. Contributions and feedback are welcome!