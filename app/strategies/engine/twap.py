import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TWAPStrategy(BaseStrategy):
    """
    Advanced TWAP (Time-Weighted Average Price) Strategy
    - Large order execution
    - Market impact minimization
    - Dynamic order sizing
    - Volume profile analysis
    - Execution monitoring
    """
    
    def __init__(self,
                 total_order_size: float = 1000,
                 execution_duration_hours: int = 4,
                 time_slices: int = 16,
                 volume_weighting: bool = True,
                 market_impact_threshold: float = 0.02,
                 max_slice_size: float = 0.1,
                 min_slice_size: float = 0.01,
                 price_deviation_threshold: float = 0.05,
                 volume_profile_periods: int = 20,
                 adaptive_sizing: bool = True):
        
        super().__init__()
        self.total_order_size = total_order_size
        self.execution_duration_hours = execution_duration_hours
        self.time_slices = time_slices
        self.volume_weighting = volume_weighting
        self.market_impact_threshold = market_impact_threshold
        self.max_slice_size = max_slice_size
        self.min_slice_size = min_slice_size
        self.price_deviation_threshold = price_deviation_threshold
        self.volume_profile_periods = volume_profile_periods
        self.adaptive_sizing = adaptive_sizing
        
        self.execution_plan = None
        self.executed_slices = []
        self.remaining_size = total_order_size
        self.start_time = None
        self.end_time = None
        
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for Bayesian optimization"""
        return {
            'total_order_size': {'type': 'real', 'min': 100, 'max': 10000},
            'execution_duration_hours': {'type': 'integer', 'min': 1, 'max': 24},
            'time_slices': {'type': 'integer', 'min': 8, 'max': 48},
            'market_impact_threshold': {'type': 'real', 'min': 0.01, 'max': 0.05},
            'max_slice_size': {'type': 'real', 'min': 0.05, 'max': 0.2},
            'min_slice_size': {'type': 'real', 'min': 0.005, 'max': 0.05},
            'price_deviation_threshold': {'type': 'real', 'min': 0.02, 'max': 0.1},
            'volume_profile_periods': {'type': 'integer', 'min': 10, 'max': 50}
        }
    
    def create_execution_plan(self, data: pd.DataFrame, side: str) -> Dict[str, Any]:
        """Create TWAP execution plan"""
        try:
            if data.empty:
                return None
            
            # Calculate time intervals
            slice_duration = self.execution_duration_hours * 3600 / self.time_slices  # seconds
            
            # Analyze volume profile
            volume_profile = self.analyze_volume_profile(data)
            
            # Create execution schedule
            execution_schedule = []
            current_time = datetime.now()
            self.start_time = current_time
            self.end_time = current_time + timedelta(hours=self.execution_duration_hours)
            
            # Calculate base slice size
            base_slice_size = self.total_order_size / self.time_slices
            
            for i in range(self.time_slices):
                slice_time = current_time + timedelta(seconds=i * slice_duration)
                
                # Get volume weight for this time period
                volume_weight = self.get_volume_weight(slice_time, volume_profile)
                
                # Calculate adaptive slice size
                if self.adaptive_sizing:
                    slice_size = base_slice_size * volume_weight
                    slice_size = max(self.min_slice_size * self.total_order_size, 
                                   min(self.max_slice_size * self.total_order_size, slice_size))
                else:
                    slice_size = base_slice_size
                
                execution_schedule.append({
                    'slice_id': i,
                    'scheduled_time': slice_time,
                    'size': slice_size,
                    'volume_weight': volume_weight,
                    'status': 'pending',
                    'executed_price': None,
                    'executed_time': None
                })
            
            self.execution_plan = {
                'side': side,
                'total_size': self.total_order_size,
                'duration_hours': self.execution_duration_hours,
                'time_slices': self.time_slices,
                'schedule': execution_schedule,
                'volume_profile': volume_profile,
                'created_at': current_time
            }
            
            return self.execution_plan
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            return None
    
    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile for optimal execution timing"""
        try:
            if len(data) < self.volume_profile_periods:
                return {'hourly_volumes': {}, 'peak_hours': [], 'low_volume_hours': []}
            
            # Calculate hourly volume averages
            data['hour'] = data.index.hour if hasattr(data.index, 'hour') else pd.to_datetime(data.index).hour
            hourly_volumes = data.groupby('hour')['volume'].mean()
            
            # Normalize volumes
            total_volume = hourly_volumes.sum()
            normalized_volumes = hourly_volumes / total_volume
            
            # Find peak and low volume hours
            volume_threshold_high = normalized_volumes.quantile(0.75)
            volume_threshold_low = normalized_volumes.quantile(0.25)
            
            peak_hours = normalized_volumes[normalized_volumes > volume_threshold_high].index.tolist()
            low_volume_hours = normalized_volumes[normalized_volumes < volume_threshold_low].index.tolist()
            
            return {
                'hourly_volumes': normalized_volumes.to_dict(),
                'peak_hours': peak_hours,
                'low_volume_hours': low_volume_hours,
                'avg_volume': total_volume / 24
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return {'hourly_volumes': {}, 'peak_hours': [], 'low_volume_hours': []}
    
    def get_volume_weight(self, time: datetime, volume_profile: Dict) -> float:
        """Get volume weight for a specific time"""
        try:
            hour = time.hour
            hourly_volumes = volume_profile.get('hourly_volumes', {})
            
            if hour in hourly_volumes:
                volume_weight = hourly_volumes[hour]
            else:
                volume_weight = 1.0 / 24  # Default equal weight
            
            # Adjust weight based on peak/low volume hours
            if hour in volume_profile.get('peak_hours', []):
                volume_weight *= 1.2  # Increase weight for peak hours
            elif hour in volume_profile.get('low_volume_hours', []):
                volume_weight *= 0.8  # Decrease weight for low volume hours
            
            return max(0.1, min(2.0, volume_weight))  # Bound between 0.1 and 2.0
            
        except Exception as e:
            logger.error(f"Error getting volume weight: {e}")
            return 1.0
    
    def calculate_market_impact(self, slice_size: float, current_volume: float, avg_volume: float) -> float:
        """Calculate expected market impact of a trade slice"""
        try:
            # Simple market impact model
            volume_ratio = slice_size / avg_volume if avg_volume > 0 else 0
            impact = volume_ratio * 0.1  # 10% impact per volume ratio
            
            return min(self.market_impact_threshold, impact)
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}")
            return 0.0
    
    def adjust_slice_size(self, slice_size: float, market_conditions: Dict) -> float:
        """Dynamically adjust slice size based on market conditions"""
        try:
            adjusted_size = slice_size
            
            # Adjust based on volatility
            volatility = market_conditions.get('volatility', 0.02)
            if volatility > 0.05:  # High volatility
                adjusted_size *= 0.8  # Reduce size
            elif volatility < 0.01:  # Low volatility
                adjusted_size *= 1.2  # Increase size
            
            # Adjust based on spread
            spread = market_conditions.get('spread', 0.001)
            if spread > 0.005:  # Wide spread
                adjusted_size *= 0.9  # Reduce size
            elif spread < 0.001:  # Tight spread
                adjusted_size *= 1.1  # Increase size
            
            # Ensure bounds
            adjusted_size = max(self.min_slice_size * self.total_order_size,
                              min(self.max_slice_size * self.total_order_size, adjusted_size))
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error adjusting slice size: {e}")
            return slice_size
    
    def execute_slice(self, slice_info: Dict, current_price: float, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Execute a single TWAP slice"""
        try:
            # Check if slice should be executed
            current_time = datetime.now()
            if current_time < slice_info['scheduled_time']:
                return {'status': 'waiting', 'reason': 'Not yet scheduled'}
            
            # Check if slice is already executed
            if slice_info['status'] == 'executed':
                return {'status': 'completed', 'reason': 'Already executed'}
            
            # Calculate market conditions
            market_conditions = self.analyze_market_conditions(market_data)
            
            # Adjust slice size if needed
            adjusted_size = self.adjust_slice_size(slice_info['size'], market_conditions)
            
            # Check market impact
            impact = self.calculate_market_impact(adjusted_size, 
                                               market_conditions.get('current_volume', 0),
                                               market_conditions.get('avg_volume', 0))
            
            if impact > self.market_impact_threshold:
                # Reduce size to minimize impact
                adjusted_size *= (self.market_impact_threshold / impact)
            
            # Check price deviation
            price_deviation = abs(current_price - self.get_vwap(market_data)) / current_price
            if price_deviation > self.price_deviation_threshold:
                return {'status': 'deferred', 'reason': f'Price deviation too high: {price_deviation:.2%}'}
            
            # Execute the slice
            execution_result = {
                'slice_id': slice_info['slice_id'],
                'executed_size': adjusted_size,
                'executed_price': current_price,
                'executed_time': current_time,
                'market_impact': impact,
                'price_deviation': price_deviation,
                'market_conditions': market_conditions
            }
            
            # Update slice info
            slice_info.update({
                'status': 'executed',
                'executed_price': current_price,
                'executed_time': current_time,
                'actual_size': adjusted_size
            })
            
            # Update remaining size
            self.remaining_size -= adjusted_size
            self.executed_slices.append(execution_result)
            
            return {'status': 'executed', 'result': execution_result}
            
        except Exception as e:
            logger.error(f"Error executing slice: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market conditions"""
        try:
            if data.empty:
                return {'volatility': 0.02, 'spread': 0.001, 'current_volume': 0, 'avg_volume': 0}
            
            # Calculate volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate spread (simplified)
            spread = (data['high'].iloc[-1] - data['low'].iloc[-1]) / data['close'].iloc[-1]
            
            # Calculate volume metrics
            current_volume = data['volume'].iloc[-1] if 'volume' in data.columns else 0
            avg_volume = data['volume'].mean() if 'volume' in data.columns else 0
            
            return {
                'volatility': volatility,
                'spread': spread,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'vwap': self.get_vwap(data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {'volatility': 0.02, 'spread': 0.001, 'current_volume': 0, 'avg_volume': 0}
    
    def get_vwap(self, data: pd.DataFrame) -> float:
        """Calculate VWAP (Volume Weighted Average Price)"""
        try:
            if data.empty or 'volume' not in data.columns:
                return data['close'].iloc[-1] if not data.empty else 0
            
            vwap = (data['close'] * data['volume']).sum() / data['volume'].sum()
            return vwap
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return data['close'].iloc[-1] if not data.empty else 0
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        try:
            if not self.execution_plan:
                return {'status': 'not_started', 'progress': 0.0}
            
            executed_slices = [s for s in self.execution_plan['schedule'] if s['status'] == 'executed']
            total_executed = sum(s.get('actual_size', s['size']) for s in executed_slices)
            
            progress = total_executed / self.total_order_size
            avg_price = np.mean([s['executed_price'] for s in executed_slices]) if executed_slices else 0
            
            return {
                'status': 'in_progress' if progress < 1.0 else 'completed',
                'progress': progress,
                'executed_slices': len(executed_slices),
                'total_slices': self.time_slices,
                'remaining_size': self.remaining_size,
                'avg_execution_price': avg_price,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'estimated_completion': self.get_estimated_completion()
            }
            
        except Exception as e:
            logger.error(f"Error getting execution status: {e}")
            return {'status': 'error', 'progress': 0.0}
    
    def get_estimated_completion(self) -> Optional[datetime]:
        """Estimate completion time based on current progress"""
        try:
            if not self.execution_plan or not self.start_time:
                return None
            
            status = self.get_execution_status()
            progress = status['progress']
            
            if progress == 0:
                return self.end_time
            
            elapsed_time = datetime.now() - self.start_time
            estimated_total_time = elapsed_time / progress
            estimated_completion = self.start_time + estimated_total_time
            
            return estimated_completion
            
        except Exception as e:
            logger.error(f"Error estimating completion: {e}")
            return None
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate TWAP execution signals"""
        if not self.execution_plan:
            return []
        
        try:
            signals = []
            current_time = datetime.now()
            
            # Check for slices that need execution
            for slice_info in self.execution_plan['schedule']:
                if slice_info['status'] == 'pending':
                    execution_result = self.execute_slice(slice_info, data['close'].iloc[-1], data)
                    
                    if execution_result['status'] == 'executed':
                        signals.append({
                            'signal': self.execution_plan['side'],
                            'price': execution_result['result']['executed_price'],
                            'timestamp': current_time,
                            'confidence': 1.0,
                            'reason': f"TWAP slice {slice_info['slice_id']} executed",
                            'size': execution_result['result']['executed_size'],
                            'market_impact': execution_result['result']['market_impact'],
                            'setup_type': 'twap_execution'
                        })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating TWAP signals: {e}")
            return []
    
    def update_position(self, signal: Dict[str, Any], account_balance: float):
        """Update position based on signal"""
        # TWAP doesn't maintain traditional positions, it executes orders
        # This method is kept for compatibility but doesn't apply to TWAP
        pass
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get latest market data"""
        # This would be implemented to fetch real-time data
        return pd.DataFrame()
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        execution_status = self.get_execution_status()
        
        return {
            'name': 'TWAP Strategy',
            'description': 'Time-Weighted Average Price execution for large orders',
            'parameters': {
                'total_order_size': self.total_order_size,
                'execution_duration_hours': self.execution_duration_hours,
                'time_slices': self.time_slices,
                'volume_weighting': self.volume_weighting,
                'market_impact_threshold': self.market_impact_threshold,
                'adaptive_sizing': self.adaptive_sizing
            },
            'execution_status': execution_status,
            'executed_slices_count': len(self.executed_slices),
            'remaining_size': self.remaining_size
        }