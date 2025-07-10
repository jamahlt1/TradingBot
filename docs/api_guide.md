# API Guide

## Overview
This guide documents all REST and WebSocket endpoints for the Advanced Trading Bot Platform. Use this as a reference for integrating, automating, or extending the platform.

---

## 1. Authentication

### Login
- **POST** `/api/auth/login`
- **Body:** `{ "username": "user", "password": "pass" }`
- **Response:** `{ "access_token": "...", "refresh_token": "..." }`

### Refresh Token
- **POST** `/api/auth/refresh`
- **Body:** `{ "refresh_token": "..." }`
- **Response:** `{ "access_token": "..." }`

### User Profile
- **GET** `/api/auth/profile`
- **Headers:** `Authorization: Bearer <token>`
- **Response:** User profile JSON

---

## 2. Account & Wallet Management

### List Accounts
- **GET** `/api/accounts`
- **Response:** List of trading accounts

### Connect Wallet
- **POST** `/api/wallets/connect`
- **Body:** `{ "type": "crypto|metatrader", "details": {...} }`
- **Response:** Wallet/account info

### Get Balances
- **GET** `/api/balances`
- **Response:** `{ "accounts": [...], "wallets": [...] }`

### Withdraw Funds
- **POST** `/api/withdraw`
- **Body:** `{ "account_id": "...", "amount": 100.0 }`
- **Response:** Withdrawal status

---

## 3. Strategy Management

### List Strategies
- **GET** `/api/strategies`
- **Response:** List of strategies

### Create Strategy
- **POST** `/api/strategies`
- **Body:** Strategy config JSON
- **Response:** Created strategy

### Update Strategy
- **PUT** `/api/strategies/{id}`
- **Body:** Partial/full config
- **Response:** Updated strategy

### Delete Strategy
- **DELETE** `/api/strategies/{id}`
- **Response:** Success/failure

### Activate/Pause/Stop
- **POST** `/api/strategies/{id}/activate|pause|stop`
- **Response:** Status

### Emergency Stop All
- **POST** `/api/strategies/emergency-stop`
- **Response:** All strategies stopped

### Upload Trading Plan
- **POST** `/api/strategies/upload`
- **Body:** File upload (JSON/CSV)
- **Response:** Import status

---

## 4. Trade Execution

### Place Order
- **POST** `/api/trades/place`
- **Body:** `{ "account_id": "...", "symbol": "BTCUSDT", "side": "buy", "type": "market|limit|stop|trailing", "quantity": 1.0, "price": 30000 }`
- **Response:** Order/trade status

### List Trades
- **GET** `/api/trades?account_id=...`
- **Response:** List of trades

### Emergency Close All
- **POST** `/api/trades/emergency-close`
- **Body:** `{ "account_id": "..." }`
- **Response:** All positions closed

---

## 5. Backtesting

### Run Backtest
- **POST** `/api/backtest/run`
- **Body:** Backtest config JSON
- **Response:** Backtest job/status

### Get Backtest Results
- **GET** `/api/backtest/{id}`
- **Response:** Backtest results, analytics, charts

---

## 6. ML/AI Endpoints

### Get AI Suggestions
- **POST** `/api/ai/suggest-strategy`
- **Body:** `{ "strategy_id": "..." }`
- **Response:** AI/LLM suggestions

### Review Strategy
- **POST** `/api/ai/review-strategy`
- **Body:** `{ "strategy_id": "..." }`
- **Response:** Full analysis, recommendations

### Optimize Parameters
- **POST** `/api/ai/optimize-parameters`
- **Body:** `{ "strategy_id": "..." }`
- **Response:** Optimized parameters

---

## 7. Notifications

### List Notifications
- **GET** `/api/notifications`
- **Response:** List of notifications

### Mark as Read
- **POST** `/api/notifications/read`
- **Body:** `{ "notification_id": "..." }`
- **Response:** Status

---

## 8. WebSocket Events

- **URL:** `ws://<host>/ws`
- **Events:**
  - `trade_update`: Real-time trade status
  - `strategy_update`: Strategy status changes
  - `balance_update`: Balance changes
  - `alert`: Failsafe or risk alert
  - `ai_recommendation`: New AI suggestion

---

## 9. Example Requests

### cURL Example
```bash
curl -X POST https://api.yourplatform.com/api/strategies \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "strategy_name": "Trend Following", ... }'
```

### Python Example
```python
import requests
headers = {"Authorization": f"Bearer {token}"}
resp = requests.get("https://api.yourplatform.com/api/strategies", headers=headers)
print(resp.json())
```

---

## 10. Error Codes
- `400`: Bad request (invalid input)
- `401`: Unauthorized (invalid/missing token)
- `403`: Forbidden (insufficient permissions)
- `404`: Not found
- `409`: Conflict (duplicate)
- `500`: Internal server error

---

## 11. Troubleshooting
- **401 Unauthorized:** Check your token and login status.
- **403 Forbidden:** You may not have access to this resource.
- **429 Too Many Requests:** Rate limit exceeded, wait and retry.
- **500 Server Error:** Contact support if persistent.

---

For more, see the [CLI Guide](cli_guide.md) and [Strategy Guide](strategy_guide.md).