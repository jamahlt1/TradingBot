import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Advanced ML
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# Feature Engineering
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

logger = logging.getLogger(__name__)

class MLModelType(Enum):
    """ML Model Types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    ENSEMBLE = "ensemble"
    ARIMA = "arima"
    SARIMA = "sarima"

@dataclass
class MLPrediction:
    """ML Prediction Result"""
    prediction: float
    confidence: float
    model_type: str
    features_used: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]

class EnhancedMLEngine:
    """
    Advanced ML Engine with OpenRouter Integration
    - Multiple ML algorithms
    - Feature engineering
    - Model ensemble
    - Real-time predictions
    - OpenRouter AI analysis
    """
    
    def __init__(self,
                 openrouter_api_key: str = None,
                 model_types: List[MLModelType] = None,
                 feature_engineering: bool = True,
                 ensemble_method: str = 'voting',
                 auto_optimization: bool = True):
        
        self.openrouter_api_key = openrouter_api_key
        self.model_types = model_types or [
            MLModelType.XGBOOST,
            MLModelType.LIGHTGBM,
            MLModelType.LSTM,
            MLModelType.ENSEMBLE
        ]
        
        self.feature_engineering = feature_engineering
        self.ensemble_method = ensemble_method
        self.auto_optimization = auto_optimization
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # OpenRouter integration
        self.openrouter_client = OpenRouterClient(api_key=openrouter_api_key)
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = []
        
    async def initialize_models(self, data: pd.DataFrame, target_column: str = 'target'):
        """Initialize all ML models"""
        try:
            logger.info("Initializing ML models...")
            
            # Feature engineering
            if self.feature_engineering:
                data = self.engineer_features(data)
            
            # Prepare data
            X, y = self.prepare_data(data, target_column)
            
            # Initialize models
            for model_type in self.model_types:
                model = self.create_model(model_type, X.shape[1])
                self.models[model_type.value] = model
                
                # Create scaler
                self.scalers[model_type.value] = StandardScaler()
                
            logger.info(f"Initialized {len(self.models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        try:
            df = data.copy()
            
            # Technical indicators
            df = self.add_technical_indicators(df)
            
            # Price-based features
            df = self.add_price_features(df)
            
            # Volume features
            df = self.add_volume_features(df)
            
            # Time-based features
            df = self.add_time_features(df)
            
            # Statistical features
            df = self.add_statistical_features(df)
            
            # Market microstructure features
            df = self.add_microstructure_features(df)
            
            # Remove NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # Trend indicators
            df['sma_5'] = SMAIndicator(close=df['close'], window=5).sma_indicator()
            df['sma_10'] = SMAIndicator(close=df['close'], window=10).sma_indicator()
            df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
            df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
            
            # MACD
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # ADX
            df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
            
            # Momentum indicators
            df['rsi'] = RSIIndicator(close=df['close']).rsi()
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
            
            # Volatility indicators
            bb = BollingerBands(close=df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
            
            # Volume indicators
            if 'volume' in df.columns:
                df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
                df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            # Returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price changes
            df['price_change'] = df['close'] - df['close'].shift(1)
            df['price_change_pct'] = df['price_change'] / df['close'].shift(1)
            
            # High-Low features
            df['hl_ratio'] = df['high'] / df['low']
            df['hl_range'] = df['high'] - df['low']
            df['hl_range_pct'] = df['hl_range'] / df['close']
            
            # Open-Close features
            df['oc_ratio'] = df['open'] / df['close']
            df['oc_range'] = df['close'] - df['open']
            df['oc_range_pct'] = df['oc_range'] / df['open']
            
            # Moving averages ratios
            df['sma_ratio_5_20'] = df['sma_5'] / df['sma_20']
            df['sma_ratio_10_50'] = df['sma_10'] / df['sma_50']
            df['ema_ratio_12_26'] = df['ema_12'] / df['ema_26']
            
            # Price position
            df['price_vs_sma20'] = df['close'] / df['sma_20']
            df['price_vs_sma50'] = df['close'] / df['sma_50']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price features: {e}")
            return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            if 'volume' not in df.columns:
                return df
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_change'] = df['volume'].pct_change()
            
            # Volume-price relationship
            df['volume_price_trend'] = df['volume'] * df['returns']
            df['volume_price_ratio'] = df['volume'] / df['close']
            
            # Volume momentum
            df['volume_momentum'] = df['volume'] - df['volume'].shift(5)
            df['volume_momentum_pct'] = df['volume_momentum'] / df['volume'].shift(5)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volume features: {e}")
            return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            # Convert index to datetime if needed
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Time features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Market session features
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            return df
    
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        try:
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'returns_mean_{window}'] = df['returns'].rolling(window=window).mean()
                df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
                df[f'returns_skew_{window}'] = df['returns'].rolling(window=window).skew()
                df[f'returns_kurt_{window}'] = df['returns'].rolling(window=window).kurt()
                
                df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()
            
            # Z-scores
            df['returns_zscore'] = (df['returns'] - df['returns'].rolling(20).mean()) / df['returns'].rolling(20).std()
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            
            # Percentile ranks
            df['returns_percentile'] = df['returns'].rolling(20).rank(pct=True)
            df['volume_percentile'] = df['volume'].rolling(20).rank(pct=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding statistical features: {e}")
            return df
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        try:
            # Bid-ask spread approximation
            df['spread_approx'] = (df['high'] - df['low']) / df['close']
            
            # Price efficiency
            df['price_efficiency'] = abs(df['close'] - df['close'].shift(1)) / df['atr']
            
            # Volatility clustering
            df['volatility_cluster'] = df['returns'].rolling(5).std() / df['returns'].rolling(20).std()
            
            # Momentum indicators
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Mean reversion indicators
            df['mean_reversion_5'] = (df['close'] - df['sma_5']) / df['sma_5']
            df['mean_reversion_20'] = (df['close'] - df['sma_20']) / df['sma_20']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding microstructure features: {e}")
            return df
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML models"""
        try:
            # Remove target column if it exists
            feature_columns = [col for col in data.columns if col != target_column]
            
            # Select features
            X = data[feature_columns].values
            y = data[target_column].values if target_column in data.columns else np.zeros(len(data))
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return np.array([]), np.array([])
    
    def create_model(self, model_type: MLModelType, n_features: int) -> Any:
        """Create ML model based on type"""
        try:
            if model_type == MLModelType.RANDOM_FOREST:
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            
            elif model_type == MLModelType.GRADIENT_BOOSTING:
                return GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
            
            elif model_type == MLModelType.XGBOOST:
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            
            elif model_type == MLModelType.LIGHTGBM:
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            
            elif model_type == MLModelType.CATBOOST:
                return CatBoostClassifier(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=False
                )
            
            elif model_type == MLModelType.SVM:
                return SVC(
                    kernel='rbf',
                    C=1.0,
                    random_state=42,
                    probability=True
                )
            
            elif model_type == MLModelType.NEURAL_NETWORK:
                return MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                )
            
            elif model_type == MLModelType.LSTM:
                return self.create_lstm_model(n_features)
            
            elif model_type == MLModelType.GRU:
                return self.create_gru_model(n_features)
            
            elif model_type == MLModelType.CNN:
                return self.create_cnn_model(n_features)
            
            elif model_type == MLModelType.ENSEMBLE:
                return self.create_ensemble_model()
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error creating model {model_type}: {e}")
            return None
    
    def create_lstm_model(self, n_features: int) -> Model:
        """Create LSTM model"""
        try:
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(None, n_features)),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            return None
    
    def create_gru_model(self, n_features: int) -> Model:
        """Create GRU model"""
        try:
            model = Sequential([
                GRU(128, return_sequences=True, input_shape=(None, n_features)),
                Dropout(0.2),
                GRU(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating GRU model: {e}")
            return None
    
    def create_cnn_model(self, n_features: int) -> Model:
        """Create CNN model"""
        try:
            model = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=(None, n_features)),
                MaxPooling1D(2),
                Conv1D(32, 3, activation='relu'),
                MaxPooling1D(2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating CNN model: {e}")
            return None
    
    def create_ensemble_model(self) -> Dict[str, Any]:
        """Create ensemble model"""
        try:
            models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42),
                'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42)
            }
            
            return {
                'models': models,
                'method': self.ensemble_method
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
            return None
    
    async def train_models(self, data: pd.DataFrame, target_column: str = 'target'):
        """Train all ML models"""
        try:
            logger.info("Training ML models...")
            
            # Feature engineering
            if self.feature_engineering:
                data = self.engineer_features(data)
            
            # Prepare data
            X, y = self.prepare_data(data, target_column)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No data available for training")
                return False
            
            # Train each model
            for model_name, model in self.models.items():
                try:
                    logger.info(f"Training {model_name}...")
                    
                    # Scale features
                    X_scaled = self.scalers[model_name].fit_transform(X)
                    
                    if model_name in ['lstm', 'gru', 'cnn']:
                        # Reshape for deep learning models
                        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                        
                        # Train deep learning model
                        model.fit(
                            X_reshaped, y,
                            epochs=50,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[
                                EarlyStopping(patience=10, restore_best_weights=True),
                                ReduceLROnPlateau(patience=5, factor=0.5)
                            ],
                            verbose=0
                        )
                    else:
                        # Train traditional ML model
                        model.fit(X_scaled, y)
                    
                    # Calculate feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[model_name] = model.feature_importances_
                    
                    logger.info(f"Trained {model_name} successfully")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    async def predict(self, data: pd.DataFrame, model_type: str = None) -> MLPrediction:
        """Make prediction using ML models"""
        try:
            # Feature engineering
            if self.feature_engineering:
                data = self.engineer_features(data)
            
            # Prepare features
            feature_columns = [col for col in data.columns if col not in ['target', 'prediction']]
            X = data[feature_columns].iloc[-1:].values
            
            if len(X) == 0:
                return MLPrediction(
                    prediction=0.0,
                    confidence=0.0,
                    model_type="none",
                    features_used=[],
                    timestamp=datetime.now(),
                    metadata={}
                )
            
            predictions = []
            confidences = []
            
            # Get predictions from all models
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    X_scaled = self.scalers[model_name].transform(X)
                    
                    if model_name in ['lstm', 'gru', 'cnn']:
                        # Reshape for deep learning models
                        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                        pred = model.predict(X_reshaped)[0][0]
                    else:
                        # Traditional ML prediction
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(X_scaled)[0][1]
                        else:
                            pred = model.predict(X_scaled)[0]
                    
                    predictions.append(pred)
                    confidences.append(abs(pred - 0.5) * 2)  # Confidence based on distance from 0.5
                    
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
                    predictions.append(0.5)
                    confidences.append(0.0)
            
            # Ensemble prediction
            if self.ensemble_method == 'voting':
                final_prediction = np.mean(predictions)
                final_confidence = np.mean(confidences)
            elif self.ensemble_method == 'weighted':
                weights = np.array(confidences) / np.sum(confidences) if np.sum(confidences) > 0 else np.ones(len(confidences)) / len(confidences)
                final_prediction = np.average(predictions, weights=weights)
                final_confidence = np.average(confidences, weights=weights)
            else:
                final_prediction = np.median(predictions)
                final_confidence = np.median(confidences)
            
            # Get OpenRouter analysis
            openrouter_analysis = await self.get_openrouter_analysis(data)
            
            return MLPrediction(
                prediction=final_prediction,
                confidence=final_confidence,
                model_type="ensemble",
                features_used=feature_columns,
                timestamp=datetime.now(),
                metadata={
                    'individual_predictions': predictions,
                    'individual_confidences': confidences,
                    'openrouter_analysis': openrouter_analysis
                }
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return MLPrediction(
                prediction=0.5,
                confidence=0.0,
                model_type="error",
                features_used=[],
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )
    
    async def get_openrouter_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get OpenRouter AI analysis"""
        try:
            if not self.openrouter_client:
                return {}
            
            # Prepare market data summary
            latest_data = data.tail(20)
            market_summary = {
                'current_price': float(latest_data['close'].iloc[-1]),
                'price_change': float(latest_data['close'].iloc[-1] - latest_data['close'].iloc[0]),
                'volume': float(latest_data['volume'].iloc[-1]) if 'volume' in latest_data.columns else 0,
                'rsi': float(latest_data['rsi'].iloc[-1]) if 'rsi' in latest_data.columns else 50,
                'macd': float(latest_data['macd'].iloc[-1]) if 'macd' in latest_data.columns else 0,
                'volatility': float(latest_data['returns'].std()) if 'returns' in latest_data.columns else 0
            }
            
            # Create analysis prompt
            prompt = f"""
            Analyze the following market data and provide trading insights:
            
            Market Data:
            - Current Price: ${market_summary['current_price']:.2f}
            - Price Change: {market_summary['price_change']:.2f}
            - RSI: {market_summary['rsi']:.2f}
            - MACD: {market_summary['macd']:.4f}
            - Volatility: {market_summary['volatility']:.4f}
            
            Please provide:
            1. Market sentiment (bullish/bearish/neutral)
            2. Key technical levels to watch
            3. Risk assessment
            4. Trading recommendations
            5. Confidence level (0-100%)
            """
            
            # Get OpenRouter analysis
            analysis = await self.openrouter_client.analyze_market(prompt)
            
            return {
                'market_summary': market_summary,
                'ai_analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Error getting OpenRouter analysis: {e}")
            return {}
    
    def evaluate_model_performance(self, data: pd.DataFrame, target_column: str = 'target') -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            # Feature engineering
            if self.feature_engineering:
                data = self.engineer_features(data)
            
            # Prepare data
            X, y = self.prepare_data(data, target_column)
            
            if len(X) == 0 or len(y) == 0:
                return {}
            
            performance_metrics = {}
            
            # Evaluate each model
            for model_name, model in self.models.items():
                try:
                    # Scale features
                    X_scaled = self.scalers[model_name].transform(X)
                    
                    if model_name in ['lstm', 'gru', 'cnn']:
                        # Reshape for deep learning models
                        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                        y_pred = model.predict(X_reshaped).flatten()
                    else:
                        # Traditional ML prediction
                        if hasattr(model, 'predict_proba'):
                            y_pred = model.predict_proba(X_scaled)[:, 1]
                        else:
                            y_pred = model.predict(X_scaled)
                    
                    # Calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(y, y_pred.round()),
                        'precision': precision_score(y, y_pred.round(), zero_division=0),
                        'recall': recall_score(y, y_pred.round(), zero_division=0),
                        'f1_score': f1_score(y, y_pred.round(), zero_division=0),
                        'roc_auc': roc_auc_score(y, y_pred),
                        'mse': mean_squared_error(y, y_pred),
                        'mae': mean_absolute_error(y, y_pred),
                        'r2': r2_score(y, y_pred)
                    }
                    
                    performance_metrics[model_name] = metrics
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    performance_metrics[model_name] = {}
            
            self.model_performance = performance_metrics
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {}
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, Any]:
        """Get feature importance for models"""
        try:
            if model_name:
                return {model_name: self.feature_importance.get(model_name, [])}
            else:
                return self.feature_importance
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary"""
        try:
            return {
                'models_trained': len(self.models),
                'model_types': [model_type.value for model_type in self.model_types],
                'feature_engineering': self.feature_engineering,
                'ensemble_method': self.ensemble_method,
                'performance_metrics': self.model_performance,
                'feature_importance': self.feature_importance,
                'prediction_history_count': len(self.prediction_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {}


class OpenRouterClient:
    """OpenRouter API client for AI analysis"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_market(self, prompt: str) -> Dict[str, Any]:
        """Analyze market using OpenRouter"""
        try:
            if not self.api_key:
                return {'error': 'No API key provided'}
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'deepseek-chat',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert financial analyst and trading advisor. Provide clear, actionable insights based on market data.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.3,
                'max_tokens': 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'analysis': result['choices'][0]['message']['content'],
                            'model': result['model'],
                            'usage': result['usage']
                        }
                    else:
                        return {'error': f'API request failed: {response.status}'}
                        
        except Exception as e:
            logger.error(f"Error in OpenRouter analysis: {e}")
            return {'error': str(e)}