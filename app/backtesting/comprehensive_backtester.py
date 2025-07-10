import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from app.ml.enhanced_ml_engine import EnhancedMLEngine, MLModelType, MLPrediction
from app.core.strategy_base import StrategyBase
from app.core.risk_manager import RiskManager
from app.core.position_manager import PositionManager

logger = logging.getLogger(__name__)

class BacktestResult(Enum):
    """Backtest Result Types"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"

@dataclass
class Trade:
    """Trade Record"""
    trade_id: str
    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    position_type: str  # "long", "short"
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    status: str  # "open", "closed", "cancelled"
    exit_reason: Optional[str]
    commission: float = 0.0
    slippage: float = 0.0
    ml_prediction: Optional[MLPrediction] = None
    technical_analysis: Optional[Dict[str, Any]] = None

@dataclass
class BacktestMetrics:
    """Comprehensive Backtest Metrics"""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percentage: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_percentage: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    treynor_ratio: float
    
    # Trade metrics
    avg_trade_duration: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    recovery_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # ML metrics
    ml_accuracy: float
    ml_prediction_correlation: float
    strategy_confidence: float
    
    # Risk-adjusted metrics
    risk_reward_ratio: float
    kelly_criterion: float
    optimal_position_size: float

@dataclass
class BacktestReport:
    """Comprehensive Backtest Report"""
    backtest_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    metrics: BacktestMetrics
    trades: List[Trade]
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    strategy_performance: Dict[str, Any]
    ml_performance: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime

class ComprehensiveBacktester:
    """
    Comprehensive Backtesting System
    - Multi-strategy backtesting
    - ML model integration
    - Advanced risk metrics
    - Detailed reporting
    - Performance visualization
    - Strategy optimization
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 ml_engine: EnhancedMLEngine = None,
                 risk_manager: RiskManager = None,
                 position_manager: PositionManager = None):
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.ml_engine = ml_engine
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        
        # Backtest state
        self.current_capital = initial_capital
        self.equity_curve = []
        self.trades = []
        self.open_positions = {}
        self.trade_counter = 0
        
        # Performance tracking
        self.daily_returns = []
        self.drawdown_series = []
        self.peak_capital = initial_capital
        
        # ML tracking
        self.ml_predictions = []
        self.ml_accuracy = []
        
    async def run_backtest(self, strategy: StrategyBase, data: pd.DataFrame, 
                          start_date: datetime = None, end_date: datetime = None,
                          parameters: Dict[str, Any] = None) -> BacktestReport:
        """Run comprehensive backtest"""
        try:
            logger.info(f"Starting backtest for {strategy.symbol}")
            
            # Initialize backtest
            self._initialize_backtest()
            
            # Filter data by date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # Update strategy parameters if provided
            if parameters:
                strategy.update_parameters(parameters)
            
            # Initialize ML engine if available
            if self.ml_engine:
                await self._initialize_ml_engine(data)
            
            # Run backtest
            for i in range(len(data)):
                current_data = data.iloc[:i+1]
                current_time = data.index[i]
                
                # Update equity curve
                self._update_equity_curve(current_time)
                
                # Execute strategy
                await self._execute_strategy_step(strategy, current_data, current_time)
                
                # Update open positions
                self._update_positions(current_data.iloc[-1], current_time)
            
            # Close remaining positions
            self._close_all_positions(data.iloc[-1], data.index[-1])
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            # Generate report
            report = self._generate_report(strategy, data, metrics)
            
            logger.info(f"Backtest completed. Final capital: ${report.final_capital:.2f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _initialize_backtest(self):
        """Initialize backtest state"""
        self.current_capital = self.initial_capital
        self.equity_curve = []
        self.trades = []
        self.open_positions = {}
        self.trade_counter = 0
        self.daily_returns = []
        self.drawdown_series = []
        self.peak_capital = self.initial_capital
        self.ml_predictions = []
        self.ml_accuracy = []
    
    async def _initialize_ml_engine(self, data: pd.DataFrame):
        """Initialize ML engine for backtesting"""
        try:
            if self.ml_engine:
                # Prepare target variable
                data_copy = data.copy()
                data_copy['target'] = data_copy['close'].shift(-1) / data_copy['close'] - 1
                data_copy['target'] = (data_copy['target'] > 0).astype(int)
                
                # Initialize and train ML models
                await self.ml_engine.initialize_models(data_copy, 'target')
                await self.ml_engine.train_models(data_copy, 'target')
                
                logger.info("ML engine initialized for backtesting")
                
        except Exception as e:
            logger.error(f"Error initializing ML engine: {e}")
    
    async def _execute_strategy_step(self, strategy: StrategyBase, data: pd.DataFrame, current_time: datetime):
        """Execute single strategy step"""
        try:
            # Get strategy signal
            signal = await strategy.execute_strategy(data)
            
            if signal['action'] in ['buy', 'sell']:
                # Get ML prediction if available
                ml_prediction = None
                if self.ml_engine:
                    ml_prediction = await self.ml_engine.predict(data)
                    self.ml_predictions.append(ml_prediction)
                
                # Calculate position size
                position_size = self._calculate_position_size(signal, ml_prediction)
                
                # Execute trade
                if signal['action'] == 'buy':
                    await self._execute_buy_trade(strategy, data.iloc[-1], current_time, position_size, ml_prediction)
                elif signal['action'] == 'sell':
                    await self._execute_sell_trade(strategy, data.iloc[-1], current_time, position_size, ml_prediction)
            
        except Exception as e:
            logger.error(f"Error executing strategy step: {e}")
    
    def _calculate_position_size(self, signal: Dict[str, Any], ml_prediction: Optional[MLPrediction] = None) -> float:
        """Calculate position size based on risk management"""
        try:
            # Base position size (2% risk per trade)
            base_risk = self.current_capital * 0.02
            
            # Adjust based on signal confidence
            confidence = signal.get('confidence', 0.5)
            position_size = base_risk * confidence
            
            # Adjust based on ML prediction
            if ml_prediction:
                ml_confidence = ml_prediction.confidence
                position_size *= (0.5 + ml_confidence)
            
            # Ensure position size doesn't exceed available capital
            max_position = self.current_capital * 0.1  # Max 10% per trade
            position_size = min(position_size, max_position)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.current_capital * 0.02
    
    async def _execute_buy_trade(self, strategy: StrategyBase, current_data: pd.Series, 
                                current_time: datetime, position_size: float, 
                                ml_prediction: Optional[MLPrediction] = None):
        """Execute buy trade"""
        try:
            # Calculate entry price with slippage
            entry_price = current_data['close'] * (1 + self.slippage_rate)
            
            # Calculate position size in units
            units = position_size / entry_price
            
            # Calculate commission
            commission = position_size * self.commission_rate
            
            # Create trade record
            trade = Trade(
                trade_id=f"trade_{self.trade_counter}",
                symbol=strategy.symbol,
                strategy=strategy.__class__.__name__,
                entry_time=current_time,
                exit_time=None,
                entry_price=entry_price,
                exit_price=None,
                position_size=position_size,
                position_type="long",
                pnl=None,
                pnl_percentage=None,
                stop_loss=entry_price * 0.95,  # 5% stop loss
                take_profit=entry_price * 1.15,  # 15% take profit
                status="open",
                exit_reason=None,
                commission=commission,
                slippage=position_size * self.slippage_rate,
                ml_prediction=ml_prediction
            )
            
            # Add to trades list
            self.trades.append(trade)
            
            # Add to open positions
            self.open_positions[trade.trade_id] = trade
            
            # Update capital
            self.current_capital -= (position_size + commission)
            
            self.trade_counter += 1
            
        except Exception as e:
            logger.error(f"Error executing buy trade: {e}")
    
    async def _execute_sell_trade(self, strategy: StrategyBase, current_data: pd.Series, 
                                 current_time: datetime, position_size: float,
                                 ml_prediction: Optional[MLPrediction] = None):
        """Execute sell trade (short)"""
        try:
            # Calculate entry price with slippage
            entry_price = current_data['close'] * (1 - self.slippage_rate)
            
            # Calculate position size in units
            units = position_size / entry_price
            
            # Calculate commission
            commission = position_size * self.commission_rate
            
            # Create trade record
            trade = Trade(
                trade_id=f"trade_{self.trade_counter}",
                symbol=strategy.symbol,
                strategy=strategy.__class__.__name__,
                entry_time=current_time,
                exit_time=None,
                entry_price=entry_price,
                exit_price=None,
                position_size=position_size,
                position_type="short",
                pnl=None,
                pnl_percentage=None,
                stop_loss=entry_price * 1.05,  # 5% stop loss
                take_profit=entry_price * 0.85,  # 15% take profit
                status="open",
                exit_reason=None,
                commission=commission,
                slippage=position_size * self.slippage_rate,
                ml_prediction=ml_prediction
            )
            
            # Add to trades list
            self.trades.append(trade)
            
            # Add to open positions
            self.open_positions[trade.trade_id] = trade
            
            # Update capital
            self.current_capital -= commission
            
            self.trade_counter += 1
            
        except Exception as e:
            logger.error(f"Error executing sell trade: {e}")
    
    def _update_positions(self, current_data: pd.Series, current_time: datetime):
        """Update open positions"""
        try:
            positions_to_close = []
            
            for trade_id, trade in self.open_positions.items():
                current_price = current_data['close']
                
                # Check stop loss
                if trade.position_type == "long":
                    if current_price <= trade.stop_loss:
                        trade.exit_reason = "stop_loss"
                        positions_to_close.append(trade_id)
                    elif current_price >= trade.take_profit:
                        trade.exit_reason = "take_profit"
                        positions_to_close.append(trade_id)
                else:  # short
                    if current_price >= trade.stop_loss:
                        trade.exit_reason = "stop_loss"
                        positions_to_close.append(trade_id)
                    elif current_price <= trade.take_profit:
                        trade.exit_reason = "take_profit"
                        positions_to_close.append(trade_id)
            
            # Close positions
            for trade_id in positions_to_close:
                self._close_position(trade_id, current_data, current_time)
                
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _close_position(self, trade_id: str, current_data: pd.Series, current_time: datetime):
        """Close a specific position"""
        try:
            trade = self.open_positions[trade_id]
            
            # Calculate exit price with slippage
            if trade.position_type == "long":
                exit_price = current_data['close'] * (1 - self.slippage_rate)
            else:
                exit_price = current_data['close'] * (1 + self.slippage_rate)
            
            # Calculate PnL
            if trade.position_type == "long":
                pnl = (exit_price - trade.entry_price) * (trade.position_size / trade.entry_price)
            else:
                pnl = (trade.entry_price - exit_price) * (trade.position_size / trade.entry_price)
            
            # Calculate commission
            exit_commission = (trade.position_size / trade.entry_price) * exit_price * self.commission_rate
            
            # Update trade
            trade.exit_time = current_time
            trade.exit_price = exit_price
            trade.pnl = pnl - trade.commission - exit_commission
            trade.pnl_percentage = trade.pnl / trade.position_size
            trade.status = "closed"
            
            # Update capital
            if trade.position_type == "long":
                self.current_capital += (trade.position_size + trade.pnl - exit_commission)
            else:
                self.current_capital += (trade.position_size + trade.pnl - exit_commission)
            
            # Remove from open positions
            del self.open_positions[trade_id]
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _close_all_positions(self, current_data: pd.Series, current_time: datetime):
        """Close all remaining positions"""
        try:
            for trade_id in list(self.open_positions.keys()):
                self._close_position(trade_id, current_data, current_time)
                
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
    
    def _update_equity_curve(self, current_time: datetime):
        """Update equity curve"""
        try:
            # Calculate current equity
            open_positions_value = sum(trade.position_size for trade in self.open_positions.values())
            current_equity = self.current_capital + open_positions_value
            
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': current_equity,
                'capital': self.current_capital,
                'open_positions': len(self.open_positions)
            })
            
            # Update peak capital
            if current_equity > self.peak_capital:
                self.peak_capital = current_equity
            
            # Calculate drawdown
            drawdown = (self.peak_capital - current_equity) / self.peak_capital
            self.drawdown_series.append(drawdown)
            
        except Exception as e:
            logger.error(f"Error updating equity curve: {e}")
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        try:
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            equity_df['returns'] = equity_df['equity'].pct_change()
            equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod()
            
            # Basic metrics
            total_trades = len([t for t in self.trades if t.status == "closed"])
            winning_trades = len([t for t in self.trades if t.status == "closed" and t.pnl > 0])
            losing_trades = len([t for t in self.trades if t.status == "closed" and t.pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # PnL metrics
            total_pnl = sum(t.pnl for t in self.trades if t.status == "closed")
            total_pnl_percentage = (self.current_capital - self.initial_capital) / self.initial_capital
            
            # Risk metrics
            max_drawdown = max(self.drawdown_series) if self.drawdown_series else 0
            max_drawdown_percentage = max_drawdown * 100
            
            # Calculate Sharpe ratio
            returns = equity_df['returns'].dropna()
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Calculate Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            # Calculate Calmar ratio
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            # Calculate VaR and CVaR
            var_95, cvar_95 = self._calculate_var_cvar(returns)
            
            # Performance metrics
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            annualized_return = self._calculate_annualized_return(equity_df)
            volatility = returns.std() * np.sqrt(252)
            
            # Trade metrics
            closed_trades = [t for t in self.trades if t.status == "closed"]
            if closed_trades:
                avg_trade_duration = np.mean([
                    (t.exit_time - t.entry_time).total_seconds() / 3600 
                    for t in closed_trades
                ])
                avg_win = np.mean([t.pnl for t in closed_trades if t.pnl > 0]) if any(t.pnl > 0 for t in closed_trades) else 0
                avg_loss = np.mean([t.pnl for t in closed_trades if t.pnl < 0]) if any(t.pnl < 0 for t in closed_trades) else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                avg_trade_duration = avg_win = avg_loss = profit_factor = 0
            
            # Consecutive wins/losses
            max_consecutive_wins = self._calculate_max_consecutive(closed_trades, lambda x: x.pnl > 0)
            max_consecutive_losses = self._calculate_max_consecutive(closed_trades, lambda x: x.pnl < 0)
            
            # ML metrics
            ml_accuracy = self._calculate_ml_accuracy()
            ml_prediction_correlation = self._calculate_ml_correlation()
            strategy_confidence = np.mean([t.ml_prediction.confidence for t in self.trades if t.ml_prediction]) if any(t.ml_prediction for t in self.trades) else 0
            
            # Risk-adjusted metrics
            risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            kelly_criterion = self._calculate_kelly_criterion(win_rate, avg_win, avg_loss)
            optimal_position_size = kelly_criterion * self.current_capital
            
            return BacktestMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pnl_percentage=total_pnl_percentage,
                max_drawdown=max_drawdown,
                max_drawdown_percentage=max_drawdown_percentage,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                var_95=var_95,
                cvar_95=cvar_95,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                beta=0,  # Would need market data
                alpha=0,  # Would need market data
                information_ratio=0,  # Would need benchmark
                treynor_ratio=0,  # Would need market data
                avg_trade_duration=avg_trade_duration,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                recovery_factor=total_return / max_drawdown if max_drawdown > 0 else 0,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                ml_accuracy=ml_accuracy,
                ml_prediction_correlation=ml_prediction_correlation,
                strategy_confidence=strategy_confidence,
                risk_reward_ratio=risk_reward_ratio,
                kelly_criterion=kelly_criterion,
                optimal_position_size=optimal_position_size
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns - risk_free_rate / 252
            return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            excess_returns = returns - risk_free_rate / 252
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std()
            return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        try:
            annualized_return = returns.mean() * 252
            return annualized_return / max_drawdown if max_drawdown > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0
    
    def _calculate_var_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        try:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            cvar = returns[returns <= var].mean()
            return var, cvar
        except Exception as e:
            logger.error(f"Error calculating VaR/CVaR: {e}")
            return 0, 0
    
    def _calculate_annualized_return(self, equity_df: pd.DataFrame) -> float:
        """Calculate annualized return"""
        try:
            total_days = (equity_df.index[-1] - equity_df.index[0]).days
            total_return = (equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]) / equity_df['equity'].iloc[0]
            return (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating annualized return: {e}")
            return 0
    
    def _calculate_max_consecutive(self, trades: List[Trade], condition) -> int:
        """Calculate maximum consecutive wins/losses"""
        try:
            max_consecutive = current_consecutive = 0
            
            for trade in trades:
                if condition(trade):
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
        except Exception as e:
            logger.error(f"Error calculating max consecutive: {e}")
            return 0
    
    def _calculate_ml_accuracy(self) -> float:
        """Calculate ML prediction accuracy"""
        try:
            if not self.ml_predictions:
                return 0
            
            correct_predictions = 0
            total_predictions = 0
            
            for trade in self.trades:
                if trade.ml_prediction and trade.status == "closed":
                    predicted_direction = trade.ml_prediction.prediction > 0.5
                    actual_direction = trade.pnl > 0
                    
                    if predicted_direction == actual_direction:
                        correct_predictions += 1
                    total_predictions += 1
            
            return correct_predictions / total_predictions if total_predictions > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating ML accuracy: {e}")
            return 0
    
    def _calculate_ml_correlation(self) -> float:
        """Calculate correlation between ML predictions and actual returns"""
        try:
            if not self.ml_predictions:
                return 0
            
            predictions = []
            returns = []
            
            for trade in self.trades:
                if trade.ml_prediction and trade.status == "closed":
                    predictions.append(trade.ml_prediction.prediction)
                    returns.append(trade.pnl_percentage)
            
            if len(predictions) > 1:
                return np.corrcoef(predictions, returns)[0, 1]
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating ML correlation: {e}")
            return 0
    
    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion"""
        try:
            if avg_loss == 0:
                return 0
            
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            return max(0, min(kelly, 1))  # Constrain between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {e}")
            return 0
    
    def _generate_report(self, strategy: StrategyBase, data: pd.DataFrame, metrics: BacktestMetrics) -> BacktestReport:
        """Generate comprehensive backtest report"""
        try:
            # Create equity curve DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Create drawdown curve
            drawdown_df = pd.DataFrame({
                'timestamp': [e['timestamp'] for e in self.equity_curve],
                'drawdown': self.drawdown_series
            })
            drawdown_df.set_index('timestamp', inplace=True)
            
            # Calculate monthly returns
            monthly_returns = equity_df['equity'].resample('M').last().pct_change()
            monthly_returns_df = pd.DataFrame({
                'month': monthly_returns.index,
                'return': monthly_returns.values
            })
            
            # Strategy performance analysis
            strategy_performance = {
                'strategy_name': strategy.__class__.__name__,
                'symbol': strategy.symbol,
                'timeframe': strategy.timeframe,
                'parameters': strategy.get_parameters() if hasattr(strategy, 'get_parameters') else {},
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown_percentage
            }
            
            # ML performance analysis
            ml_performance = {
                'ml_accuracy': metrics.ml_accuracy,
                'prediction_correlation': metrics.ml_prediction_correlation,
                'strategy_confidence': metrics.strategy_confidence,
                'total_predictions': len(self.ml_predictions),
                'avg_prediction_confidence': np.mean([p.confidence for p in self.ml_predictions]) if self.ml_predictions else 0
            }
            
            # Risk analysis
            risk_analysis = {
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'kelly_criterion': metrics.kelly_criterion,
                'optimal_position_size': metrics.optimal_position_size
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)
            
            return BacktestReport(
                backtest_id=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy_name=strategy.__class__.__name__,
                symbol=strategy.symbol,
                timeframe=strategy.timeframe,
                start_date=data.index[0],
                end_date=data.index[-1],
                initial_capital=self.initial_capital,
                final_capital=self.current_capital,
                metrics=metrics,
                trades=self.trades,
                equity_curve=equity_df,
                drawdown_curve=drawdown_df,
                monthly_returns=monthly_returns_df,
                strategy_performance=strategy_performance,
                ml_performance=ml_performance,
                risk_analysis=risk_analysis,
                recommendations=recommendations,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def _generate_recommendations(self, metrics: BacktestMetrics) -> List[str]:
        """Generate trading recommendations based on backtest results"""
        recommendations = []
        
        try:
            # Win rate recommendations
            if metrics.win_rate < 0.4:
                recommendations.append("Consider improving entry criteria - win rate is below 40%")
            elif metrics.win_rate > 0.6:
                recommendations.append("Excellent win rate - consider increasing position sizes")
            
            # Risk management recommendations
            if metrics.max_drawdown_percentage > 20:
                recommendations.append("Reduce position sizes to limit drawdown")
            
            if metrics.sharpe_ratio < 1.0:
                recommendations.append("Strategy needs improvement - Sharpe ratio below 1.0")
            
            # Profit factor recommendations
            if metrics.profit_factor < 1.5:
                recommendations.append("Improve risk-reward ratio - profit factor below 1.5")
            
            # ML recommendations
            if metrics.ml_accuracy < 0.5:
                recommendations.append("ML model needs retraining - accuracy below 50%")
            
            # Kelly criterion recommendations
            if metrics.kelly_criterion < 0.1:
                recommendations.append("Consider reducing position sizes based on Kelly criterion")
            elif metrics.kelly_criterion > 0.3:
                recommendations.append("Consider increasing position sizes based on Kelly criterion")
            
            # General recommendations
            if metrics.total_trades < 30:
                recommendations.append("Need more trades for statistical significance")
            
            if metrics.avg_trade_duration < 1:
                recommendations.append("Consider longer holding periods to reduce transaction costs")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def generate_visualizations(self, report: BacktestReport) -> Dict[str, Any]:
        """Generate comprehensive visualizations"""
        try:
            visualizations = {}
            
            # Equity curve
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=report.equity_curve.index,
                y=report.equity_curve['equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue')
            ))
            fig_equity.update_layout(
                title='Portfolio Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                showlegend=True
            )
            visualizations['equity_curve'] = fig_equity
            
            # Drawdown chart
            fig_drawdown = go.Figure()
            fig_drawdown.add_trace(go.Scatter(
                x=report.drawdown_curve.index,
                y=report.drawdown_curve['drawdown'] * 100,
                mode='lines',
                name='Drawdown (%)',
                line=dict(color='red'),
                fill='tonexty'
            ))
            fig_drawdown.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                showlegend=True
            )
            visualizations['drawdown'] = fig_drawdown
            
            # Monthly returns heatmap
            if len(report.monthly_returns) > 0:
                monthly_returns_pivot = report.monthly_returns.set_index('month')['return'].unstack()
                fig_monthly = px.imshow(
                    monthly_returns_pivot.values,
                    x=monthly_returns_pivot.columns,
                    y=monthly_returns_pivot.index,
                    title='Monthly Returns Heatmap',
                    color_continuous_scale='RdYlGn'
                )
                visualizations['monthly_returns'] = fig_monthly
            
            # Trade distribution
            if report.trades:
                trade_pnls = [t.pnl for t in report.trades if t.status == "closed"]
                fig_distribution = go.Figure()
                fig_distribution.add_trace(go.Histogram(
                    x=trade_pnls,
                    nbinsx=20,
                    name='Trade PnL Distribution'
                ))
                fig_distribution.update_layout(
                    title='Trade PnL Distribution',
                    xaxis_title='PnL ($)',
                    yaxis_title='Frequency'
                )
                visualizations['trade_distribution'] = fig_distribution
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return {}
    
    def export_report(self, report: BacktestReport, format: str = "json") -> str:
        """Export backtest report"""
        try:
            if format == "json":
                # Convert report to JSON-serializable format
                report_dict = {
                    'backtest_id': report.backtest_id,
                    'strategy_name': report.strategy_name,
                    'symbol': report.symbol,
                    'timeframe': report.timeframe,
                    'start_date': report.start_date.isoformat(),
                    'end_date': report.end_date.isoformat(),
                    'initial_capital': report.initial_capital,
                    'final_capital': report.final_capital,
                    'metrics': asdict(report.metrics),
                    'strategy_performance': report.strategy_performance,
                    'ml_performance': report.ml_performance,
                    'risk_analysis': report.risk_analysis,
                    'recommendations': report.recommendations,
                    'generated_at': report.generated_at.isoformat()
                }
                
                return json.dumps(report_dict, indent=2)
            
            elif format == "csv":
                # Export trades to CSV
                trades_data = []
                for trade in report.trades:
                    trades_data.append({
                        'trade_id': trade.trade_id,
                        'symbol': trade.symbol,
                        'strategy': trade.strategy,
                        'entry_time': trade.entry_time.isoformat(),
                        'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'position_size': trade.position_size,
                        'position_type': trade.position_type,
                        'pnl': trade.pnl,
                        'pnl_percentage': trade.pnl_percentage,
                        'status': trade.status,
                        'exit_reason': trade.exit_reason
                    })
                
                df = pd.DataFrame(trades_data)
                return df.to_csv(index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            raise