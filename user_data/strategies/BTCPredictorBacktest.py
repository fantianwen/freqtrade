"""
BTC Predictor Backtest Strategy

A backtest-compatible version of BTCPredictorStrategy that:
1. Loads the trained ML model directly
2. Calculates features from historical OHLCV data
3. Uses informative pairs for multi-timeframe support
4. Implements cumulative confidence logic

Usage:
    freqtrade backtesting --strategy BTCPredictorBacktest \
        --config user_data/config_btc_predictor.json \
        --timerange 20231201-20241201
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
from collections import deque

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, informative

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical indicators calculator - matching prediction_server.py"""
    
    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
        low_n = df['low'].rolling(window=n).min()
        high_n = df['high'].rolling(window=n).max()
        
        rsv = (df['close'] - low_n) / (high_n - low_n) * 100
        rsv = rsv.fillna(50)
        
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return k, d, j
    
    @staticmethod
    def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = TechnicalIndicators.calculate_ema(close, fast)
        ema_slow = TechnicalIndicators.calculate_ema(close, slow)
        
        macd = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd, signal)
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    @staticmethod
    def detect_crossover(fast: pd.Series, slow: pd.Series) -> Tuple[pd.Series, pd.Series]:
        golden = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        death = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        return golden, death
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi = rsi.fillna(50)
        rsi = rsi.replace([np.inf, -np.inf], 50)
        
        return rsi
    
    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 10) -> pd.Series:
        roc = (close - close.shift(period)) / close.shift(period) * 100
        return roc.fillna(0)
    
    @staticmethod
    def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
        mom = close - close.shift(period)
        return mom.fillna(0)
    
    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_n = df['high'].rolling(window=period).max()
        low_n = df['low'].rolling(window=period).min()
        
        wr = (high_n - df['close']) / (high_n - low_n) * -100
        return wr.fillna(-50)
    
    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        cci = (tp - sma_tp) / (0.015 * mad)
        cci = cci.fillna(0)
        cci = cci.replace([np.inf, -np.inf], 0)
        
        return cci
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        adx = adx.fillna(25).replace([np.inf, -np.inf], 25)
        plus_di = plus_di.fillna(25).replace([np.inf, -np.inf], 25)
        minus_di = minus_di.fillna(25).replace([np.inf, -np.inf], 25)
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> Tuple[pd.Series, pd.Series]:
        rsi = TechnicalIndicators.calculate_rsi(close, rsi_period)
        
        rsi_low = rsi.rolling(window=stoch_period).min()
        rsi_high = rsi.rolling(window=stoch_period).max()
        
        stoch_rsi_k = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
        stoch_rsi_d = stoch_rsi_k.rolling(window=3).mean()
        
        stoch_rsi_k = stoch_rsi_k.fillna(50)
        stoch_rsi_d = stoch_rsi_d.fillna(50)
        
        return stoch_rsi_k, stoch_rsi_d


class CumulativeConfidenceTracker:
    """Track predictions and calculate cumulative confidence"""
    
    def __init__(self, ewma_alpha: float = 0.3, consistency_window: int = 4,
                 boost_factor: float = 1.1, penalty_factor: float = 0.9):
        self.ewma_alpha = ewma_alpha
        self.consistency_window = consistency_window
        self.boost_factor = boost_factor
        self.penalty_factor = penalty_factor
        self.history = deque(maxlen=20)  # Keep last 20 predictions
    
    def add_prediction(self, direction: str, confidence: float, prediction_pct: float) -> float:
        """Add prediction and return cumulative confidence"""
        
        # Calculate EWMA
        if len(self.history) == 0:
            ewma = confidence
        else:
            ewma = confidence
            for pred in list(self.history):
                ewma = self.ewma_alpha * ewma + (1 - self.ewma_alpha) * pred['confidence']
        
        # Direction consistency check
        recent = list(self.history)[-self.consistency_window:]
        recent_dirs = [p['direction'] for p in recent] + [direction]
        recent_dirs = recent_dirs[-self.consistency_window:]
        
        unique_dirs = set(recent_dirs)
        
        if len(unique_dirs) == 1 and direction != '震荡':
            ewma = min(100, ewma * self.boost_factor)
        elif len(unique_dirs) > 2 or '震荡' in unique_dirs:
            ewma = ewma * self.penalty_factor
        
        cumulative = round(ewma, 2)
        
        self.history.append({
            'direction': direction,
            'confidence': confidence,
            'prediction_pct': prediction_pct,
            'cumulative_confidence': cumulative
        })
        
        return cumulative
    
    def get_current_cumulative(self) -> float:
        if len(self.history) == 0:
            return 0.0
        return self.history[-1]['cumulative_confidence']
    
    def reset(self):
        self.history.clear()


class BTCPredictorBacktest(IStrategy):
    """
    Backtest-compatible BTC Predictor Strategy with direct model loading.
    """
    
    INTERFACE_VERSION = 3
    can_short: bool = True
    
    # Use 1h as primary timeframe (model was trained on 1h features)
    timeframe = "1h"
    
    # ROI disabled - use exit signals
    minimal_roi = {"0": 100}
    
    stoploss = -0.05
    
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True
    
    # Need enough candles for indicators
    startup_candle_count: int = 100
    
    # Configuration
    model_path: str = ""
    entry_threshold_pct: float = 2.0
    confidence_threshold: float = 75.0
    ewma_alpha: float = 0.3
    consistency_window: int = 4
    
    # Internal state
    _model = None
    _scaler = None
    _feature_names = None
    _confidence_tracker: Optional[CumulativeConfidenceTracker] = None
    _indicators = TechnicalIndicators()
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        predictor_config = config.get('predictor_config', {})
        
        # Model path - try multiple locations
        self.model_path = predictor_config.get('model_path', '')
        if not self.model_path:
            # Default model paths to try
            possible_paths = [
                '../volitality-prediction-tool/project/models/regression_model_20251213_203325.pkl',
                'user_data/models/regression_model.pkl',
                '../models/regression_model.pkl',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_path = path
                    break
        
        self.entry_threshold_pct = predictor_config.get('entry_threshold_pct', 2.0)
        self.confidence_threshold = predictor_config.get('confidence_threshold', 75.0)
        self.ewma_alpha = predictor_config.get('ewma_alpha', 0.3)
        self.consistency_window = predictor_config.get('consistency_window', 4)
        
        # Initialize confidence tracker
        self._confidence_tracker = CumulativeConfidenceTracker(
            ewma_alpha=self.ewma_alpha,
            consistency_window=self.consistency_window
        )
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained ML model"""
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")
            logger.warning("Backtest will use simplified prediction logic")
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different model formats
            if isinstance(model_data, dict):
                if 'best_model' in model_data:
                    self._model = model_data['best_model']
                    self._scaler = model_data.get('scaler')
                    self._feature_names = model_data.get('feature_names', [])
                elif 'model' in model_data:
                    self._model = model_data['model']
                    self._scaler = model_data.get('scaler')
                    self._feature_names = model_data.get('feature_names', [])
            else:
                self._model = model_data
                self._feature_names = getattr(model_data, 'feature_names_in_', [])
            
            logger.info(f"Model loaded from {self.model_path}")
            logger.info(f"Model type: {type(self._model).__name__}")
            logger.info(f"Features expected: {len(self._feature_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
    
    def _extract_features_for_row(self, df: DataFrame, idx: int) -> Dict[str, float]:
        """Extract features for a single row using historical data up to that point"""
        
        # Get data up to this point (at least 50 rows for indicators)
        start_idx = max(0, idx - 100)
        hist_df = df.iloc[start_idx:idx+1].copy()
        
        if len(hist_df) < 50:
            return {}
        
        features = {}
        tf = '1h'  # Primary timeframe
        
        try:
            # KDJ
            k, d, j = self._indicators.calculate_kdj(hist_df)
            
            # MACD
            macd, signal, hist_macd = self._indicators.calculate_macd(hist_df['close'])
            
            # RSI
            rsi_7 = self._indicators.calculate_rsi(hist_df['close'], period=7)
            rsi_14 = self._indicators.calculate_rsi(hist_df['close'], period=14)
            rsi_21 = self._indicators.calculate_rsi(hist_df['close'], period=21)
            
            # Crossover signals
            kdj_golden, kdj_death = self._indicators.detect_crossover(k, d)
            macd_golden, macd_death = self._indicators.detect_crossover(macd, signal)
            
            # Volatility
            returns = hist_df['close'].pct_change()
            volatility = returns.std() * np.sqrt(24 * 365)  # Annualized for 1h
            
            # Volume features
            volume = hist_df['volume']
            volume_ma5 = volume.rolling(5).mean()
            volume_ma10 = volume.rolling(10).mean()
            volume_ma20 = volume.rolling(20).mean()
            
            vol_ratio_ma5 = volume.iloc[-1] / volume_ma5.iloc[-1] if volume_ma5.iloc[-1] > 0 else 1
            vol_ratio_ma10 = volume.iloc[-1] / volume_ma10.iloc[-1] if volume_ma10.iloc[-1] > 0 else 1
            vol_ratio_ma20 = volume.iloc[-1] / volume_ma20.iloc[-1] if volume_ma20.iloc[-1] > 0 else 1
            
            vol_change_1 = (volume.iloc[-1] - volume.iloc[-2]) / volume.iloc[-2] * 100 if volume.iloc[-2] > 0 else 0
            vol_change_5 = (volume.iloc[-1] - volume.iloc[-6]) / volume.iloc[-6] * 100 if len(volume) > 5 and volume.iloc[-6] > 0 else 0
            
            vol_trend = (volume_ma5.iloc[-1] - volume_ma20.iloc[-1]) / volume_ma20.iloc[-1] * 100 if volume_ma20.iloc[-1] > 0 else 0
            
            vol_high_20 = volume.tail(20).max()
            vol_low_20 = volume.tail(20).min()
            vol_position = (volume.iloc[-1] - vol_low_20) / (vol_high_20 - vol_low_20) if vol_high_20 > vol_low_20 else 0.5
            
            vol_spike = 1 if vol_ratio_ma20 > 2 else 0
            vol_shrink = 1 if vol_ratio_ma20 < 0.5 else 0
            
            price_up = 1 if hist_df['close'].iloc[-1] > hist_df['close'].iloc[-2] else 0
            vol_up = 1 if volume.iloc[-1] > volume.iloc[-2] else 0
            vol_price_divergence = 1 if price_up != vol_up else 0
            
            # Price position
            recent_high = hist_df['high'].tail(20).max()
            recent_low = hist_df['low'].tail(20).min()
            price_position = (hist_df['close'].iloc[-1] - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            
            # Trend strength
            ma20 = hist_df['close'].rolling(20).mean()
            trend_strength = (hist_df['close'].iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1] * 100 if ma20.iloc[-1] > 0 else 0
            
            # RSI derived
            rsi_14_value = rsi_14.iloc[-1]
            rsi_overbought = 1 if rsi_14_value > 70 else 0
            rsi_oversold = 1 if rsi_14_value < 30 else 0
            rsi_trend = rsi_14.iloc[-1] - rsi_14.iloc[-5] if len(rsi_14) > 5 else 0
            
            # Momentum indicators
            roc_5 = self._indicators.calculate_roc(hist_df['close'], period=5)
            roc_10 = self._indicators.calculate_roc(hist_df['close'], period=10)
            roc_20 = self._indicators.calculate_roc(hist_df['close'], period=20)
            
            mom_10 = self._indicators.calculate_momentum(hist_df['close'], period=10)
            mom_20 = self._indicators.calculate_momentum(hist_df['close'], period=20)
            
            williams_r = self._indicators.calculate_williams_r(hist_df, period=14)
            cci = self._indicators.calculate_cci(hist_df, period=20)
            adx, plus_di, minus_di = self._indicators.calculate_adx(hist_df, period=14)
            stoch_rsi_k, stoch_rsi_d = self._indicators.calculate_stoch_rsi(hist_df['close'])
            
            cci_value = cci.iloc[-1]
            cci_overbought = 1 if cci_value > 100 else 0
            cci_oversold = 1 if cci_value < -100 else 0
            
            adx_value = adx.iloc[-1]
            adx_strong_trend = 1 if adx_value > 25 else 0
            adx_weak_trend = 1 if adx_value < 20 else 0
            trend_bullish = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else 0
            
            # Build feature dict
            features = {
                f'{tf}_kdj_k': k.iloc[-1],
                f'{tf}_kdj_d': d.iloc[-1],
                f'{tf}_kdj_j': j.iloc[-1],
                f'{tf}_kdj_golden': int(kdj_golden.iloc[-1]) if not pd.isna(kdj_golden.iloc[-1]) else 0,
                f'{tf}_kdj_death': int(kdj_death.iloc[-1]) if not pd.isna(kdj_death.iloc[-1]) else 0,
                
                f'{tf}_macd': macd.iloc[-1],
                f'{tf}_macd_signal': signal.iloc[-1],
                f'{tf}_macd_hist': hist_macd.iloc[-1],
                f'{tf}_macd_golden': int(macd_golden.iloc[-1]) if not pd.isna(macd_golden.iloc[-1]) else 0,
                f'{tf}_macd_death': int(macd_death.iloc[-1]) if not pd.isna(macd_death.iloc[-1]) else 0,
                
                f'{tf}_volatility': volatility,
                
                f'{tf}_vol_ratio_ma5': vol_ratio_ma5,
                f'{tf}_vol_ratio_ma10': vol_ratio_ma10,
                f'{tf}_vol_ratio_ma20': vol_ratio_ma20,
                f'{tf}_vol_change_1': vol_change_1,
                f'{tf}_vol_change_5': vol_change_5,
                f'{tf}_vol_trend': vol_trend,
                f'{tf}_vol_position': vol_position,
                f'{tf}_vol_spike': vol_spike,
                f'{tf}_vol_shrink': vol_shrink,
                f'{tf}_vol_price_divergence': vol_price_divergence,
                
                f'{tf}_price_position': price_position,
                f'{tf}_trend_strength': trend_strength,
                
                f'{tf}_rsi_7': rsi_7.iloc[-1],
                f'{tf}_rsi_14': rsi_14.iloc[-1],
                f'{tf}_rsi_21': rsi_21.iloc[-1],
                f'{tf}_rsi_overbought': rsi_overbought,
                f'{tf}_rsi_oversold': rsi_oversold,
                f'{tf}_rsi_trend': rsi_trend,
                
                f'{tf}_roc_5': roc_5.iloc[-1],
                f'{tf}_roc_10': roc_10.iloc[-1],
                f'{tf}_roc_20': roc_20.iloc[-1],
                
                f'{tf}_mom_10': mom_10.iloc[-1],
                f'{tf}_mom_20': mom_20.iloc[-1],
                
                f'{tf}_williams_r': williams_r.iloc[-1],
                
                f'{tf}_cci': cci.iloc[-1],
                f'{tf}_cci_overbought': cci_overbought,
                f'{tf}_cci_oversold': cci_oversold,
                
                f'{tf}_adx': adx.iloc[-1],
                f'{tf}_plus_di': plus_di.iloc[-1],
                f'{tf}_minus_di': minus_di.iloc[-1],
                f'{tf}_adx_strong_trend': adx_strong_trend,
                f'{tf}_adx_weak_trend': adx_weak_trend,
                f'{tf}_trend_bullish': trend_bullish,
                
                f'{tf}_stoch_rsi_k': stoch_rsi_k.iloc[-1],
                f'{tf}_stoch_rsi_d': stoch_rsi_d.iloc[-1],
            }
            
            # Multi-timeframe features (using 1h data only for backtest simplicity)
            golden_count = features.get(f'{tf}_kdj_golden', 0) + features.get(f'{tf}_macd_golden', 0)
            death_count = features.get(f'{tf}_kdj_death', 0) + features.get(f'{tf}_macd_death', 0)
            
            features['multi_tf_golden_count'] = golden_count
            features['multi_tf_death_count'] = death_count
            features['signal_strength'] = golden_count - death_count
            
            # Time features
            if 'date' in hist_df.columns:
                dt = pd.to_datetime(hist_df['date'].iloc[-1])
            else:
                dt = datetime.now()
            features['hour'] = dt.hour if hasattr(dt, 'hour') else 12
            features['day_of_week'] = dt.weekday() if hasattr(dt, 'weekday') else 0
            features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0
            
            # Clean features
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    features[key] = 0
            
        except Exception as e:
            logger.debug(f"Feature extraction error at idx {idx}: {e}")
            return {}
        
        return features
    
    def _predict(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        """
        Make prediction using the model or fallback logic.
        
        Returns: (prediction_pct, confidence, direction)
        """
        if self._model is None:
            # Fallback: use simple indicator-based prediction
            return self._simple_prediction(features)
        
        try:
            # Prepare features for model
            if self._feature_names:
                feature_values = [features.get(name, 0) for name in self._feature_names]
                X = pd.DataFrame([feature_values], columns=self._feature_names)
            else:
                X = pd.DataFrame([features])
            
            # Fill missing features with 0
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Scale if scaler available
            if self._scaler is not None:
                try:
                    X_scaled = self._scaler.transform(X)
                except:
                    X_scaled = X.values
            else:
                X_scaled = X.values
            
            # Predict
            prediction_pct = float(self._model.predict(X_scaled)[0])
            
            # Calculate confidence based on prediction magnitude and indicators
            confidence = self._calculate_confidence(features, prediction_pct)
            
            # Determine direction
            if prediction_pct > 0.5:
                direction = '看涨'
            elif prediction_pct < -0.5:
                direction = '看跌'
            else:
                direction = '震荡'
            
            return prediction_pct, confidence, direction
            
        except Exception as e:
            logger.debug(f"Model prediction error: {e}")
            return self._simple_prediction(features)
    
    def _simple_prediction(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        """Fallback simple prediction based on indicators"""
        
        rsi = features.get('1h_rsi_14', 50)
        macd_hist = features.get('1h_macd_hist', 0)
        trend_strength = features.get('1h_trend_strength', 0)
        signal_strength = features.get('signal_strength', 0)
        
        # Simple prediction logic
        score = 0
        score += (50 - rsi) / 50 * 2  # RSI contribution
        score += np.sign(macd_hist) * min(abs(macd_hist) / 100, 1) * 1.5
        score += trend_strength / 5
        score += signal_strength * 0.5
        
        prediction_pct = np.clip(score, -5, 5)
        confidence = min(90, max(10, 50 + abs(prediction_pct) * 8))
        
        if prediction_pct > 0.5:
            direction = '看涨'
        elif prediction_pct < -0.5:
            direction = '看跌'
        else:
            direction = '震荡'
        
        return prediction_pct, confidence, direction
    
    def _calculate_confidence(self, features: Dict[str, float], prediction_pct: float) -> float:
        """Calculate prediction confidence"""
        
        confidence = 50.0
        
        # ADX strength
        adx = features.get('1h_adx', 25)
        if adx > 25:
            confidence += 10
        elif adx < 20:
            confidence -= 10
        
        # RSI confirmation
        rsi = features.get('1h_rsi_14', 50)
        if prediction_pct > 0 and rsi < 70:
            confidence += 5
        elif prediction_pct < 0 and rsi > 30:
            confidence += 5
        
        # Trend alignment
        trend_bullish = features.get('1h_trend_bullish', 0)
        if (prediction_pct > 0 and trend_bullish) or (prediction_pct < 0 and not trend_bullish):
            confidence += 10
        
        # Prediction magnitude
        if abs(prediction_pct) > 2:
            confidence += 5
        
        # Signal strength
        signal_strength = features.get('signal_strength', 0)
        if (prediction_pct > 0 and signal_strength > 0) or (prediction_pct < 0 and signal_strength < 0):
            confidence += 5
        
        return np.clip(confidence, 10, 90)
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators and model predictions for each candle.
        """
        logger.info(f"Populating indicators for {metadata['pair']}, {len(dataframe)} candles")
        
        # Reset confidence tracker for new backtest run
        self._confidence_tracker.reset()
        
        # Initialize columns
        dataframe['pred_pct'] = 0.0
        dataframe['pred_confidence'] = 0.0
        dataframe['pred_cumulative_conf'] = 0.0
        dataframe['pred_direction'] = '震荡'
        
        # Process each candle
        for idx in range(self.startup_candle_count, len(dataframe)):
            features = self._extract_features_for_row(dataframe, idx)
            
            if features:
                pred_pct, confidence, direction = self._predict(features)
                cumulative = self._confidence_tracker.add_prediction(direction, confidence, pred_pct)
                
                dataframe.loc[dataframe.index[idx], 'pred_pct'] = pred_pct
                dataframe.loc[dataframe.index[idx], 'pred_confidence'] = confidence
                dataframe.loc[dataframe.index[idx], 'pred_cumulative_conf'] = cumulative
                dataframe.loc[dataframe.index[idx], 'pred_direction'] = direction
        
        logger.info(f"Indicators populated. Sample predictions: {dataframe['pred_pct'].tail(5).tolist()}")
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signals based on predictions.
        """
        # Long entry: 大涨 with high cumulative confidence
        dataframe.loc[
            (
                (dataframe['pred_direction'] == '看涨') &
                (dataframe['pred_pct'].abs() > self.entry_threshold_pct) &
                (dataframe['pred_cumulative_conf'] >= self.confidence_threshold) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1
        
        # Short entry: 大跌 with high cumulative confidence
        dataframe.loc[
            (
                (dataframe['pred_direction'] == '看跌') &
                (dataframe['pred_pct'].abs() > self.entry_threshold_pct) &
                (dataframe['pred_cumulative_conf'] >= self.confidence_threshold) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals when cumulative confidence drops.
        """
        # Exit when confidence drops below threshold
        dataframe.loc[
            (
                (dataframe['pred_cumulative_conf'] < self.confidence_threshold) &
                (dataframe['pred_cumulative_conf'].shift(1) >= self.confidence_threshold) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1
        
        dataframe.loc[
            (
                (dataframe['pred_cumulative_conf'] < self.confidence_threshold) &
                (dataframe['pred_cumulative_conf'].shift(1) >= self.confidence_threshold) &
                (dataframe['volume'] > 0)
            ),
            'exit_short'
        ] = 1
        
        return dataframe
    
    def informative_pairs(self):
        return []
