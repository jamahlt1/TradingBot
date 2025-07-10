# CLI Guide

## Overview
This guide explains how to use the command-line interface (CLI) for all features of the Advanced Trading Bot Platform. The CLI allows full automation, scripting, and real-time monitoring.

---

## 1. Installation & Setup

### Install CLI
```bash
pip install tradingbot-cli
```

### Initialize & Configure
```bash
tradingbot-cli init
# Follow prompts for API URL, credentials, and default account
```

---

## 2. Authentication & Profile

### Login
```bash
tradingbot-cli login --username <user> --password <pass>
```

### Show Profile
```bash
tradingbot-cli profile show
```

---

## 3. Account & Wallet Management

### List Accounts
```bash
tradingbot-cli account list
```

### Connect Wallet
```bash
tradingbot-cli wallet connect --type crypto --details '{...}'
```

### Show Balances
```bash
tradingbot-cli balance show
```

### Withdraw Funds
```bash
tradingbot-cli withdraw --account <id> --amount 100
```

---

## 4. Strategy Management

### List Strategies
```bash
tradingbot-cli strategy list
```

### Create Strategy
```bash
tradingbot-cli strategy create --config strategy.json
```

### Edit Strategy
```bash
tradingbot-cli strategy edit --id <id> --param risk_per_trade=0.02
```

### Activate/Pause/Stop
```bash
tradingbot-cli strategy activate --id <id>
tradingbot-cli strategy pause --id <id>
tradingbot-cli strategy stop --id <id>
```

### Emergency Stop All
```bash
tradingbot-cli strategy stop-all
```

### Upload Trading Plan
```bash
tradingbot-cli strategy upload --file myplan.json
```

---

## 5. Trading & Orders

### Place Order
```bash
tradingbot-cli trade place --account <id> --symbol BTCUSDT --side buy --type market --quantity 1
```

### List Trades
```bash
tradingbot-cli trade list --account <id>
```

### Emergency Close All
```bash
tradingbot-cli trade close-all --account <id>
```

---

## 6. Backtesting & Analytics

### Run Backtest
```bash
tradingbot-cli backtest run --config backtest.json
```

### Get Backtest Results
```bash
tradingbot-cli backtest results --id <id>
```

---

## 7. AI/ML & Optimization

### Get AI Suggestions
```bash
tradingbot-cli ai suggest --strategy <id>
```

### Review Strategy
```bash
tradingbot-cli ai review --strategy <id>
```

### Optimize Parameters
```bash
tradingbot-cli ai optimize --strategy <id>
```

---

## 8. Monitoring & Dashboard

### Real-Time Dashboard
```bash
tradingbot-cli dashboard
```

### Show Notifications
```bash
tradingbot-cli notify list
```

---

## 9. Scripting & Automation

- All CLI commands can be scripted in bash, Python, or any shell.
- Use `--json` flag for machine-readable output.
- Example: Run daily backtest and email results
```bash
tradingbot-cli backtest run --config mybacktest.json --json | mail -s "Backtest Results" user@example.com
```

---

## 10. Emergency & Failsafe

### Emergency Stop
```bash
tradingbot-cli stop-all
```

### Emergency Close All Trades
```bash
tradingbot-cli trade close-all --account <id>
```

---

## 11. Troubleshooting
- Use `--help` with any command for usage info.
- Check API URL and credentials if you get 401/403 errors.
- Use `tradingbot-cli logs` for recent CLI activity.
- For more, see the [API Guide](api_guide.md) and [Strategy Guide](strategy_guide.md).