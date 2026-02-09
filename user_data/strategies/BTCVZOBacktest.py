"""
BTC VZO Backtest Strategy
==========================

A freqtrade backtest strategy that combines:
1. VZO (Volume Zone Oscillator) calculated from multi-TF (5m, 15m, 30m, 1h, 4h)
2. VZO slope & slope zero-cross detection with rolling z-score normalization
3. VZO-slope divergence detection
4. ML model predictions (GBM) using 1h features + multi-TF VZO/slope
5. Cumulative confidence tracking

The strategy uses 1m as the base timeframe and fetches 5m/15m/30m/1h/4h
informative data for VZO/slope calculation and ML features.

Usage:
    # First download data:
    freqtrade download-data --config user_data/config_btc_vzo_backtest.json \
        --timeframes 5m 15m 1h 4h 1d --timerange 20240101- --trading-mode futures

    # Run backtest:
    freqtrade backtesting --strategy BTCVZOBacktest \
        --config user_data/config_btc_vzo_backtest.json \
        --timerange 20240601-20250201 \
        --export trades
"""

import logging
import os
import pickle
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, informative, merge_informative_pair

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Technical Indicators (matching prediction_server.py)
# ──────────────────────────────────────────────────────────────────────

class TechnicalIndicators:
    """Technical indicators calculator – identical to prediction_server.py"""

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3,
                      m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
        low_n = df['low'].rolling(window=n).min()
        high_n = df['high'].rolling(window=n).max()
        rsv = ((df['close'] - low_n) / (high_n - low_n) * 100).fillna(50)
        k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
        d = k.ewm(alpha=1 / m2, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    @staticmethod
    def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = TechnicalIndicators.calculate_ema(close, fast)
        ema_slow = TechnicalIndicators.calculate_ema(close, slow)
        macd = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd, signal)
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def detect_crossover(fast: pd.Series,
                         slow: pd.Series) -> Tuple[pd.Series, pd.Series]:
        golden = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        death = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        return golden, death

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).replace([np.inf, -np.inf], 50)

    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 10) -> pd.Series:
        return ((close - close.shift(period)) / close.shift(period) * 100).fillna(0)

    @staticmethod
    def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
        return (close - close.shift(period)).fillna(0)

    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_n = df['high'].rolling(window=period).max()
        low_n = df['low'].rolling(window=period).min()
        return ((high_n - df['close']) / (high_n - low_n) * -100).fillna(-50)

    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def calculate_adx(df: pd.DataFrame,
                      period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        adx = adx.fillna(25).replace([np.inf, -np.inf], 25)
        plus_di = plus_di.fillna(25).replace([np.inf, -np.inf], 25)
        minus_di = minus_di.fillna(25).replace([np.inf, -np.inf], 25)
        return adx, plus_di, minus_di

    @staticmethod
    def calculate_stoch_rsi(close: pd.Series, rsi_period: int = 14,
                            stoch_period: int = 14) -> Tuple[pd.Series, pd.Series]:
        rsi = TechnicalIndicators.calculate_rsi(close, rsi_period)
        rsi_low = rsi.rolling(window=stoch_period).min()
        rsi_high = rsi.rolling(window=stoch_period).max()
        stoch_rsi_k = ((rsi - rsi_low) / (rsi_high - rsi_low) * 100).fillna(50)
        stoch_rsi_d = stoch_rsi_k.rolling(window=3).mean().fillna(50)
        return stoch_rsi_k, stoch_rsi_d


# ──────────────────────────────────────────────────────────────────────
# VZO Indicator Functions (matching prediction_server.py exactly)
# ──────────────────────────────────────────────────────────────────────

def calculate_vzo(df: pd.DataFrame, period: int = 14,
                  ma_len: int = 9) -> Tuple[pd.Series, pd.Series]:
    """
    Volume Zone Oscillator.
    Returns (vzo, vzo_ma) series.
    """
    signed_vol = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
    vp = pd.Series(signed_vol, index=df.index).ewm(span=period, adjust=False).mean()
    tv = df['volume'].ewm(span=period, adjust=False).mean()
    vzo = (100 * vp / tv).fillna(0).replace([np.inf, -np.inf], 0)
    vzo_ma = vzo.ewm(span=ma_len, adjust=False).mean()
    return vzo, vzo_ma


def calculate_vzo_slope(vzo: pd.Series, lookback: int = 5) -> pd.Series:
    """Linear-regression slope of VZO over *lookback* periods."""
    x = np.arange(lookback, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _lr_slope(window):
        if len(window) < lookback:
            return np.nan
        y = window.values
        return ((x - x_mean) * (y - y.mean())).sum() / x_var

    return vzo.rolling(window=lookback).apply(_lr_slope, raw=False).fillna(0)


# ──────────────────────────────────────────────────────────────────────
# Cumulative Confidence Tracker (same as BTCPredictorBacktest)
# ──────────────────────────────────────────────────────────────────────

class CumulativeConfidenceTracker:
    def __init__(self, ewma_alpha: float = 0.3, consistency_window: int = 4,
                 boost_factor: float = 1.1, penalty_factor: float = 0.9):
        self.ewma_alpha = ewma_alpha
        self.consistency_window = consistency_window
        self.boost_factor = boost_factor
        self.penalty_factor = penalty_factor
        self.history: deque = deque(maxlen=20)

    def add_prediction(self, direction: str, confidence: float,
                       prediction_pct: float) -> float:
        if len(self.history) == 0:
            ewma = confidence
        else:
            ewma = confidence
            for pred in list(self.history):
                ewma = self.ewma_alpha * ewma + (1 - self.ewma_alpha) * pred['confidence']

        recent = [p['direction'] for p in list(self.history)][-self.consistency_window:] + [direction]
        unique_dirs = set(recent[-self.consistency_window:])
        if len(unique_dirs) == 1 and direction != '震荡':
            ewma = min(100, ewma * self.boost_factor)
        elif len(unique_dirs) > 2 or '震荡' in unique_dirs:
            ewma = ewma * self.penalty_factor

        cumulative = round(ewma, 2)
        self.history.append({
            'direction': direction,
            'confidence': confidence,
            'prediction_pct': prediction_pct,
            'cumulative_confidence': cumulative,
        })
        return cumulative

    def reset(self):
        self.history.clear()


# ──────────────────────────────────────────────────────────────────────
# ██  Strategy
# ──────────────────────────────────────────────────────────────────────

class BTCVZOBacktest(IStrategy):
    """
    Backtest strategy combining VZO/slope momentum signals with ML
    prediction confidence.  Uses 1m as the base timeframe and merges
    15m VZO data plus 1h ML features.

    Entry:  ML direction + big move + high confidence + VZO zone + slope direction
    Exit:   ML cumulative confidence drops below exit_confidence_threshold
    """

    INTERFACE_VERSION = 3
    can_short: bool = True

    # Primary timeframe – 1m
    timeframe = "1m"

    # ROI disabled – rely on exit signals
    minimal_roi = {"0": 100}

    stoploss = -0.05

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True

    # 1m candles – need more startup bars (500 = ~8 hours)
    startup_candle_count: int = 500

    # ── Tuneable parameters ──────────────────────────────────────────
    vzo_period: int = 14
    vzo_ma_len: int = 9
    slope_lookback: int = 5
    slope_threshold: float = 3.0          # VZO slope threshold for entry

    entry_threshold_pct: float = 2.0      # ML predicted move must exceed this %
    confidence_threshold: float = 80.0    # ML cumulative confidence for entry
    exit_confidence_threshold: float = 55.0  # exit when ML cum-conf drops below this
    ewma_alpha: float = 0.3
    consistency_window: int = 4

    # ── Internal state ───────────────────────────────────────────────
    model_path: str = ""
    _model = None
    _scaler = None
    _feature_names: list = []
    _confidence_tracker: Optional[CumulativeConfidenceTracker] = None
    _indicators = TechnicalIndicators()

    # ================================================================
    # Initialisation
    # ================================================================

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        predictor_config = config.get('predictor_config', {})

        # Model path
        self.model_path = predictor_config.get('model_path', '')
        if not self.model_path:
            for path in [
                '../volitality-prediction-tool/project/models/regression_model_20251213_203325.pkl',
                'user_data/models/regression_model.pkl',
                '../models/regression_model.pkl',
            ]:
                if os.path.exists(path):
                    self.model_path = path
                    break

        # Tuneable overrides from config
        self.entry_threshold_pct = predictor_config.get('entry_threshold_pct', 2.0)
        self.confidence_threshold = predictor_config.get('confidence_threshold', 75.0)
        self.exit_confidence_threshold = predictor_config.get('exit_confidence_threshold', 55.0)
        self.ewma_alpha = predictor_config.get('ewma_alpha', 0.3)
        self.consistency_window = predictor_config.get('consistency_window', 4)
        self.vzo_period = predictor_config.get('vzo_period', 14)
        self.vzo_ma_len = predictor_config.get('vzo_ma_len', 9)
        self.slope_lookback = predictor_config.get('slope_lookback', 5)
        self.slope_threshold = predictor_config.get('slope_threshold', 1.0)

        self._confidence_tracker = CumulativeConfidenceTracker(
            ewma_alpha=self.ewma_alpha,
            consistency_window=self.consistency_window,
        )

        self._load_model()

    def _load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning(f"Model not found at: {self.model_path}")
            logger.warning("Will use indicator-only prediction fallback.")
            return
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            if isinstance(model_data, dict):
                self._model = model_data.get('best_model') or model_data.get('model')
                self._scaler = model_data.get('scaler')
                self._feature_names = model_data.get('feature_names', [])
            else:
                self._model = model_data
                self._feature_names = list(getattr(model_data, 'feature_names_in_', []))
            logger.info(f"Model loaded: {type(self._model).__name__}, "
                        f"{len(self._feature_names)} features")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self._model = None

    # ================================================================
    # Informative pairs – fetch 15m and 1h data
    # ================================================================

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative = []
        for pair in pairs:
            for tf in ['5m', '15m', '30m', '1h', '4h']:
                informative.append((pair, tf))
        return informative

    # ================================================================
    # VZO indicators on any timeframe informative data
    # ================================================================

    def _populate_vzo(self, dataframe: DataFrame) -> DataFrame:
        """Calculate VZO, slope, z-scores, crosses, divergence on any timeframe candles."""
        vzo, vzo_ma = calculate_vzo(dataframe, self.vzo_period, self.vzo_ma_len)
        dataframe['vzo'] = vzo
        dataframe['vzo_ma'] = vzo_ma

        slope = calculate_vzo_slope(vzo, self.slope_lookback)
        dataframe['vzo_slope'] = slope
        dataframe['vzo_slope_prev'] = slope.shift(1)

        # Slope acceleration (2nd derivative)
        dataframe['slope_accel'] = slope.diff().fillna(0)

        # Per-timeframe rolling z-score normalization (window=20)
        vzo_roll_mean = vzo.rolling(20).mean()
        vzo_roll_std = vzo.rolling(20).std().replace(0, 1)
        dataframe['vzo_zscore'] = ((vzo - vzo_roll_mean) / vzo_roll_std).fillna(0).replace(
            [np.inf, -np.inf], 0)

        slope_roll_mean = slope.rolling(20).mean()
        slope_roll_std = slope.rolling(20).std().replace(0, 1)
        dataframe['slope_zscore'] = ((slope - slope_roll_mean) / slope_roll_std).fillna(0).replace(
            [np.inf, -np.inf], 0)

        # VZO zone classification: -2 (strong bear) to +2 (strong bull)
        dataframe['vzo_zone'] = 0
        dataframe.loc[vzo > 40, 'vzo_zone'] = 2
        dataframe.loc[(vzo > 15) & (vzo <= 40), 'vzo_zone'] = 1
        dataframe.loc[(vzo >= -15) & (vzo <= 15), 'vzo_zone'] = 0
        dataframe.loc[(vzo >= -40) & (vzo < -15), 'vzo_zone'] = -1
        dataframe.loc[vzo < -40, 'vzo_zone'] = -2

        # Slope direction sign (+1/0/-1)
        dataframe['slope_sign'] = 0
        dataframe.loc[slope > 0, 'slope_sign'] = 1
        dataframe.loc[slope < 0, 'slope_sign'] = -1

        # Slope zero-cross
        dataframe['slope_cross_bull'] = (
            (dataframe['vzo_slope_prev'] <= 0) & (dataframe['vzo_slope'] > 0)
        ).astype(int)
        dataframe['slope_cross_bear'] = (
            (dataframe['vzo_slope_prev'] >= 0) & (dataframe['vzo_slope'] < 0)
        ).astype(int)

        # VZO-slope divergence (vectorised approximation)
        vzo_rising = (vzo > vzo.shift(1)) & (vzo.shift(1) > vzo.shift(2))
        slope_falling = (slope < slope.shift(1)) & (slope.shift(1) < slope.shift(2))
        vzo_falling = (vzo < vzo.shift(1)) & (vzo.shift(1) < vzo.shift(2))
        slope_rising = (slope > slope.shift(1)) & (slope.shift(1) > slope.shift(2))

        dataframe['div_bearish'] = (vzo_rising & slope_falling).astype(int)
        dataframe['div_bullish'] = (vzo_falling & slope_rising).astype(int)

        # Consecutive negative / positive slope bars
        neg_streak = 0
        pos_streak = 0
        neg_list = []
        pos_list = []
        for s in slope:
            if s < 0:
                neg_streak += 1
                pos_streak = 0
            elif s > 0:
                pos_streak += 1
                neg_streak = 0
            else:
                neg_streak = 0
                pos_streak = 0
            neg_list.append(neg_streak)
            pos_list.append(pos_streak)
        dataframe['neg_slope_streak'] = neg_list
        dataframe['pos_slope_streak'] = pos_list

        return dataframe

    # ================================================================
    # ML feature extraction (1h data + multi-TF VZO/slope)
    # ================================================================

    def _extract_tf_technical_features(self, df: DataFrame, tf: str) -> dict:
        """
        Vectorised technical indicator extraction for a single timeframe.
        Returns dict of {column_name: pd.Series}.
        """
        k, d, j = self._indicators.calculate_kdj(df)
        macd, signal, hist_macd = self._indicators.calculate_macd(df['close'])
        rsi_7 = self._indicators.calculate_rsi(df['close'], 7)
        rsi_14 = self._indicators.calculate_rsi(df['close'], 14)
        rsi_21 = self._indicators.calculate_rsi(df['close'], 21)
        kdj_golden, kdj_death = self._indicators.detect_crossover(k, d)
        macd_golden, macd_death = self._indicators.detect_crossover(macd, signal)

        returns = df['close'].pct_change()
        # Annualize volatility based on timeframe
        periods_map = {'5m': 288*365, '15m': 96*365, '30m': 48*365,
                       '1h': 24*365, '4h': 6*365, '1d': 365}
        ann_factor = np.sqrt(periods_map.get(tf, 24*365))
        volatility = returns.rolling(24).std() * ann_factor

        vol = df['volume']
        vol_ma5 = vol.rolling(5).mean()
        vol_ma10 = vol.rolling(10).mean()
        vol_ma20 = vol.rolling(20).mean()
        vol_ratio_ma5 = (vol / vol_ma5).fillna(1).replace([np.inf, -np.inf], 1)
        vol_ratio_ma10 = (vol / vol_ma10).fillna(1).replace([np.inf, -np.inf], 1)
        vol_ratio_ma20 = (vol / vol_ma20).fillna(1).replace([np.inf, -np.inf], 1)
        vol_change_1 = vol.pct_change(1).fillna(0).replace([np.inf, -np.inf], 0) * 100
        vol_change_5 = vol.pct_change(5).fillna(0).replace([np.inf, -np.inf], 0) * 100

        vol_trend = ((vol_ma5 - vol_ma20) / vol_ma20 * 100).fillna(0).replace([np.inf, -np.inf], 0)
        vol_high_20 = vol.rolling(20).max()
        vol_low_20 = vol.rolling(20).min()
        vol_position = ((vol - vol_low_20) / (vol_high_20 - vol_low_20)).fillna(0.5)
        vol_spike = (vol_ratio_ma20 > 2).astype(int)
        vol_shrink = (vol_ratio_ma20 < 0.5).astype(int)
        price_up = (df['close'] > df['close'].shift(1)).astype(int)
        vol_up = (vol > vol.shift(1)).astype(int)
        vol_price_div = (price_up != vol_up).astype(int)

        recent_high = df['high'].rolling(20).max()
        recent_low = df['low'].rolling(20).min()
        price_position = ((df['close'] - recent_low) / (recent_high - recent_low)).fillna(0.5)

        ma20 = df['close'].rolling(20).mean()
        trend_strength = ((df['close'] - ma20) / ma20 * 100).fillna(0).replace([np.inf, -np.inf], 0)

        rsi_overbought = (rsi_14 > 70).astype(int)
        rsi_oversold = (rsi_14 < 30).astype(int)
        rsi_trend = (rsi_14 - rsi_14.shift(5)).fillna(0)

        roc_5 = self._indicators.calculate_roc(df['close'], 5)
        roc_10 = self._indicators.calculate_roc(df['close'], 10)
        roc_20 = self._indicators.calculate_roc(df['close'], 20)
        mom_10 = self._indicators.calculate_momentum(df['close'], 10)
        mom_20 = self._indicators.calculate_momentum(df['close'], 20)
        williams_r = self._indicators.calculate_williams_r(df, 14)
        cci = self._indicators.calculate_cci(df, 20)
        adx, plus_di, minus_di = self._indicators.calculate_adx(df, 14)
        stoch_rsi_k, stoch_rsi_d = self._indicators.calculate_stoch_rsi(df['close'])

        cci_overbought = (cci > 100).astype(int)
        cci_oversold = (cci < -100).astype(int)
        adx_strong = (adx > 25).astype(int)
        adx_weak = (adx < 20).astype(int)
        trend_bullish = (plus_di > minus_di).astype(int)

        return {
            f'{tf}_kdj_k': k, f'{tf}_kdj_d': d, f'{tf}_kdj_j': j,
            f'{tf}_kdj_golden': kdj_golden.astype(int).fillna(0),
            f'{tf}_kdj_death': kdj_death.astype(int).fillna(0),
            f'{tf}_macd': macd, f'{tf}_macd_signal': signal, f'{tf}_macd_hist': hist_macd,
            f'{tf}_macd_golden': macd_golden.astype(int).fillna(0),
            f'{tf}_macd_death': macd_death.astype(int).fillna(0),
            f'{tf}_volatility': volatility.fillna(0),
            f'{tf}_vol_ratio_ma5': vol_ratio_ma5,
            f'{tf}_vol_ratio_ma10': vol_ratio_ma10,
            f'{tf}_vol_ratio_ma20': vol_ratio_ma20,
            f'{tf}_vol_change_1': vol_change_1,
            f'{tf}_vol_change_5': vol_change_5,
            f'{tf}_vol_trend': vol_trend,
            f'{tf}_vol_position': vol_position,
            f'{tf}_vol_spike': vol_spike,
            f'{tf}_vol_shrink': vol_shrink,
            f'{tf}_vol_price_divergence': vol_price_div,
            f'{tf}_price_position': price_position,
            f'{tf}_trend_strength': trend_strength,
            f'{tf}_rsi_7': rsi_7, f'{tf}_rsi_14': rsi_14, f'{tf}_rsi_21': rsi_21,
            f'{tf}_rsi_overbought': rsi_overbought,
            f'{tf}_rsi_oversold': rsi_oversold,
            f'{tf}_rsi_trend': rsi_trend,
            f'{tf}_roc_5': roc_5, f'{tf}_roc_10': roc_10, f'{tf}_roc_20': roc_20,
            f'{tf}_mom_10': mom_10, f'{tf}_mom_20': mom_20,
            f'{tf}_williams_r': williams_r,
            f'{tf}_cci': cci,
            f'{tf}_cci_overbought': cci_overbought,
            f'{tf}_cci_oversold': cci_oversold,
            f'{tf}_adx': adx, f'{tf}_plus_di': plus_di, f'{tf}_minus_di': minus_di,
            f'{tf}_adx_strong_trend': adx_strong,
            f'{tf}_adx_weak_trend': adx_weak,
            f'{tf}_trend_bullish': trend_bullish,
            f'{tf}_stoch_rsi_k': stoch_rsi_k,
            f'{tf}_stoch_rsi_d': stoch_rsi_d,
        }

    def _extract_vzo_features(self, df: DataFrame, tf: str) -> dict:
        """
        Vectorised VZO/slope feature extraction with rolling z-score normalization.
        Returns dict of {column_name: pd.Series}.
        """
        vzo, vzo_ma = calculate_vzo(df, self.vzo_period, self.vzo_ma_len)
        slope = calculate_vzo_slope(vzo, self.slope_lookback)
        slope_accel = slope.diff().fillna(0)

        # Rolling z-score normalization (window=20)
        vzo_roll_mean = vzo.rolling(20).mean()
        vzo_roll_std = vzo.rolling(20).std().replace(0, 1)
        vzo_zscore = ((vzo - vzo_roll_mean) / vzo_roll_std).fillna(0).replace(
            [np.inf, -np.inf], 0)

        slope_roll_mean = slope.rolling(20).mean()
        slope_roll_std = slope.rolling(20).std().replace(0, 1)
        slope_zscore = ((slope - slope_roll_mean) / slope_roll_std).fillna(0).replace(
            [np.inf, -np.inf], 0)

        # VZO zone: -2 (strong bear) to +2 (strong bull)
        vzo_zone = pd.Series(0, index=df.index)
        vzo_zone = vzo_zone.where(~(vzo > 40), 2)
        vzo_zone = vzo_zone.where(~((vzo > 15) & (vzo <= 40)), 1)
        vzo_zone = vzo_zone.where(~((vzo >= -40) & (vzo < -15)), -1)
        vzo_zone = vzo_zone.where(~(vzo < -40), -2)

        # Slope sign
        slope_sign = pd.Series(0, index=df.index)
        slope_sign = slope_sign.where(~(slope > 0), 1)
        slope_sign = slope_sign.where(~(slope < 0), -1)

        return {
            f'{tf}_vzo': vzo,
            f'{tf}_vzo_ma': vzo_ma,
            f'{tf}_vzo_slope': slope,
            f'{tf}_vzo_zscore': vzo_zscore,
            f'{tf}_slope_zscore': slope_zscore,
            f'{tf}_slope_accel': slope_accel,
            f'{tf}_vzo_zone': vzo_zone,
            f'{tf}_slope_sign': slope_sign,
        }

    def _extract_1h_features_vectorised(self, df_1h: DataFrame,
                                         vzo_data: Dict[str, DataFrame]) -> DataFrame:
        """
        Vectorised feature extraction on 1h dataframe + multi-TF VZO/slope.

        Args:
            df_1h: The 1h OHLCV dataframe (used as the index/base).
            vzo_data: dict of {timeframe: dataframe} for VZO feature TFs.

        Returns feature DataFrame aligned to df_1h index.
        """
        all_features: dict = {}

        # ── 1h technical indicators (as before) ──
        tech_1h = self._extract_tf_technical_features(df_1h, '1h')
        all_features.update(tech_1h)

        # ── VZO/slope features for ALL timeframes ──
        # 1h VZO computed directly from df_1h
        vzo_1h = self._extract_vzo_features(df_1h, '1h')
        all_features.update(vzo_1h)

        # Other timeframes: merge VZO columns into 1h index via merge_asof
        for tf, tf_df in vzo_data.items():
            if tf == '1h' or len(tf_df) == 0:
                continue
            vzo_feats = self._extract_vzo_features(tf_df, tf)
            # Build a small dataframe with date + vzo features
            vzo_tf_df = pd.DataFrame(vzo_feats, index=tf_df.index)
            vzo_tf_df['date'] = tf_df['date'] if 'date' in tf_df.columns else tf_df.index

            # Merge to 1h index using merge_asof (forward-fill from lower TF)
            base = pd.DataFrame({'date': df_1h['date'] if 'date' in df_1h.columns else df_1h.index},
                                index=df_1h.index)
            base['date'] = pd.to_datetime(base['date'])
            vzo_tf_df['date'] = pd.to_datetime(vzo_tf_df['date'])
            merged = pd.merge_asof(
                base.sort_values('date'),
                vzo_tf_df.sort_values('date'),
                on='date',
                direction='backward',
            )
            # Re-index to df_1h
            merged.index = df_1h.index
            for col in vzo_feats.keys():
                all_features[col] = merged[col].fillna(0).replace([np.inf, -np.inf], 0)

        # ── Cross-timeframe VZO consensus features ──
        vzo_tfs = ['5m', '15m', '30m', '1h', '4h']
        available_tfs = [t for t in vzo_tfs if f'{t}_vzo_zone' in all_features]

        if len(available_tfs) >= 2:
            # Stack zone and sign series
            zone_stack = pd.DataFrame({t: all_features[f'{t}_vzo_zone'] for t in available_tfs},
                                      index=df_1h.index)
            sign_stack = pd.DataFrame({t: all_features[f'{t}_slope_sign'] for t in available_tfs},
                                      index=df_1h.index)

            all_features['vzo_multi_tf_bullish'] = (zone_stack > 0).sum(axis=1)
            all_features['vzo_multi_tf_bearish'] = (zone_stack < 0).sum(axis=1)
            all_features['vzo_multi_tf_consensus'] = (
                all_features['vzo_multi_tf_bullish'] - all_features['vzo_multi_tf_bearish'])
            all_features['slope_multi_tf_bullish'] = (sign_stack > 0).sum(axis=1)
            all_features['slope_multi_tf_bearish'] = (sign_stack < 0).sum(axis=1)
            all_features['slope_multi_tf_consensus'] = (
                all_features['slope_multi_tf_bullish'] - all_features['slope_multi_tf_bearish'])

            # Short vs long-term divergence
            short_tfs = [t for t in ['5m', '15m'] if t in available_tfs]
            long_tfs = [t for t in ['1h', '4h'] if t in available_tfs]
            if short_tfs and long_tfs:
                short_vzo = pd.DataFrame(
                    {t: all_features[f'{t}_vzo'] for t in short_tfs}, index=df_1h.index).mean(axis=1)
                long_vzo = pd.DataFrame(
                    {t: all_features[f'{t}_vzo'] for t in long_tfs}, index=df_1h.index).mean(axis=1)
                all_features['vzo_short_long_diff'] = short_vzo - long_vzo

                short_slope = pd.DataFrame(
                    {t: all_features[f'{t}_vzo_slope'] for t in short_tfs}, index=df_1h.index).mean(axis=1)
                long_slope = pd.DataFrame(
                    {t: all_features[f'{t}_vzo_slope'] for t in long_tfs}, index=df_1h.index).mean(axis=1)
                all_features['slope_short_long_diff'] = short_slope - long_slope

        # ── Cross signals + time features (1h based) ──
        golden_count = all_features['1h_kdj_golden'] + all_features['1h_macd_golden']
        death_count = all_features['1h_kdj_death'] + all_features['1h_macd_death']
        all_features['multi_tf_golden_count'] = golden_count
        all_features['multi_tf_death_count'] = death_count
        all_features['signal_strength'] = golden_count - death_count

        if 'date' in df_1h.columns:
            dt = pd.to_datetime(df_1h['date'])
            hour = dt.dt.hour
            dow = dt.dt.weekday
        else:
            hour = pd.Series(12, index=df_1h.index)
            dow = pd.Series(0, index=df_1h.index)
        all_features['hour'] = hour
        all_features['day_of_week'] = dow
        all_features['is_weekend'] = (dow >= 5).astype(int)

        feature_df = pd.DataFrame(all_features, index=df_1h.index)
        feature_df = feature_df.fillna(0).replace([np.inf, -np.inf], 0)

        return feature_df

    def _batch_predict(self, feature_df: DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Run predictions for every row. Returns (pred_pct, confidence, direction).
        """
        n = len(feature_df)
        pred_pct = pd.Series(0.0, index=feature_df.index)
        confidence = pd.Series(50.0, index=feature_df.index)
        direction = pd.Series('震荡', index=feature_df.index)

        if self._model is not None and self._feature_names:
            try:
                X = feature_df.reindex(columns=self._feature_names, fill_value=0)
                X = X.fillna(0).replace([np.inf, -np.inf], 0)
                if self._scaler is not None:
                    try:
                        X_scaled = self._scaler.transform(X)
                    except Exception:
                        X_scaled = X.values
                else:
                    X_scaled = X.values
                raw_pred = self._model.predict(X_scaled)
                pred_pct = pd.Series(raw_pred, index=feature_df.index, dtype=float)
            except Exception as e:
                logger.error(f"Batch predict failed: {e}")

        # Confidence from indicators
        adx_col = feature_df.get('1h_adx', pd.Series(25.0, index=feature_df.index))
        rsi_col = feature_df.get('1h_rsi_14', pd.Series(50.0, index=feature_df.index))
        trend_bull = feature_df.get('1h_trend_bullish', pd.Series(0, index=feature_df.index))
        sig_str = feature_df.get('signal_strength', pd.Series(0, index=feature_df.index))

        conf = pd.Series(50.0, index=feature_df.index)
        conf = conf + np.where(adx_col > 25, 10, np.where(adx_col < 20, -10, 0))
        conf = conf + np.where((pred_pct > 0) & (rsi_col < 70), 5,
                               np.where((pred_pct < 0) & (rsi_col > 30), 5, 0))
        conf = conf + np.where(
            ((pred_pct > 0) & (trend_bull == 1)) | ((pred_pct < 0) & (trend_bull == 0)),
            10, 0)
        conf = conf + np.where(pred_pct.abs() > 2, 5, 0)
        conf = conf + np.where(
            ((pred_pct > 0) & (sig_str > 0)) | ((pred_pct < 0) & (sig_str < 0)),
            5, 0)
        confidence = conf.clip(10, 90)

        direction = pd.Series('震荡', index=feature_df.index)
        direction = direction.where(~(pred_pct > 0.5), '看涨')
        direction = direction.where(~(pred_pct < -0.5), '看跌')

        return pred_pct, confidence, direction

    # ================================================================
    # populate_indicators – main entry point
    # ================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        logger.info(f"Populating indicators for {pair}, {len(dataframe)} candles @ {self.timeframe}")

        # ── Collect all informative timeframes ──
        vzo_timeframes = ['5m', '15m', '30m', '1h', '4h']
        vzo_data: Dict[str, DataFrame] = {}  # raw dataframes for VZO feature extraction

        if self.dp:
            for tf in vzo_timeframes:
                inf_tf = self.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                if len(inf_tf) > 0:
                    vzo_data[tf] = inf_tf.copy()

        # ── 15m informative: VZO/slope for entry/exit signal columns ──
        if '15m' in vzo_data and len(vzo_data['15m']) > 0:
            inf_15m = self._populate_vzo(vzo_data['15m'].copy())
            vzo_cols = ['date', 'vzo', 'vzo_ma', 'vzo_slope', 'vzo_slope_prev',
                        'vzo_zscore', 'slope_zscore', 'vzo_zone', 'slope_sign',
                        'slope_cross_bull', 'slope_cross_bear', 'slope_accel',
                        'div_bearish', 'div_bullish',
                        'neg_slope_streak', 'pos_slope_streak']
            inf_15m_slim = inf_15m[vzo_cols].copy()
            dataframe = merge_informative_pair(
                dataframe, inf_15m_slim,
                self.timeframe, '15m',
                ffill=True, append_timeframe=True,
            )
            logger.info(f"Merged 15m VZO data ({len(inf_15m)} rows)")

        # ── 1h informative: ML features & predictions (with multi-TF VZO) ──
        if '1h' in vzo_data and len(vzo_data['1h']) > 0:
            inf_1h = vzo_data['1h'].copy()
            feature_df = self._extract_1h_features_vectorised(inf_1h, vzo_data)
            pred_pct, confidence, direction = self._batch_predict(feature_df)
            inf_1h['ml_pred_pct'] = pred_pct.values
            inf_1h['ml_confidence'] = confidence.values
            inf_1h['ml_direction'] = direction.values

            # Cumulative confidence (sequential – cannot vectorise)
            self._confidence_tracker.reset()
            cum_conf = []
            for i in range(len(inf_1h)):
                cc = self._confidence_tracker.add_prediction(
                    inf_1h['ml_direction'].iloc[i],
                    inf_1h['ml_confidence'].iloc[i],
                    inf_1h['ml_pred_pct'].iloc[i],
                )
                cum_conf.append(cc)
            inf_1h['ml_cum_conf'] = cum_conf

            ml_cols = ['date', 'ml_pred_pct', 'ml_confidence', 'ml_cum_conf', 'ml_direction']
            inf_1h_slim = inf_1h[ml_cols].copy()
            dataframe = merge_informative_pair(
                dataframe, inf_1h_slim,
                self.timeframe, '1h',
                ffill=True, append_timeframe=True,
            )
            logger.info(f"Merged 1h ML prediction data ({len(inf_1h)} rows)")

        # ── Fill any NaN from merging ──
        for col in dataframe.columns:
            if col.startswith(('vzo', 'slope', 'div_', 'neg_', 'pos_', 'ml_')):
                dataframe[col] = dataframe[col].fillna(0)

        logger.info(f"Indicators done. Columns: {[c for c in dataframe.columns if 'vzo' in c or 'ml_' in c]}")
        return dataframe

    # ================================================================
    # Entry signals
    # ================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Column names after merge (append_timeframe=True adds "_15m" / "_1h")
        vzo = dataframe.get('vzo_15m', pd.Series(0, index=dataframe.index))
        slope = dataframe.get('vzo_slope_15m', pd.Series(0, index=dataframe.index))

        ml_pred = dataframe.get('ml_pred_pct_1h', pd.Series(0, index=dataframe.index))
        ml_conf = dataframe.get('ml_cum_conf_1h', pd.Series(0, index=dataframe.index))
        ml_dir = dataframe.get('ml_direction_1h', pd.Series('震荡', index=dataframe.index))

        thr = self.slope_threshold  # 1.0

        # ── LONG entry ───────────────────────────────────────────────
        # ML says 看涨 + big move + high confidence + slope > 3
        dataframe.loc[
            (
                (ml_dir == '看涨') &
                (ml_pred.abs() > self.entry_threshold_pct) &
                (ml_conf >= self.confidence_threshold) &
                (slope > 3) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        # ── SHORT entry ──────────────────────────────────────────────
        # ML says 看跌 + big move + high confidence + slope < -3
        dataframe.loc[
            (
                (ml_dir == '看跌') &
                (ml_pred.abs() > self.entry_threshold_pct) &
                (ml_conf >= self.confidence_threshold) &
                (slope < -3) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'
        ] = 1

        # Log signal counts
        n_long = dataframe['enter_long'].sum() if 'enter_long' in dataframe.columns else 0
        n_short = dataframe['enter_short'].sum() if 'enter_short' in dataframe.columns else 0
        logger.info(f"Entry signals – LONG: {n_long}, SHORT: {n_short}")

        return dataframe

    # ================================================================
    # Exit signals
    # ================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ml_conf = dataframe.get('ml_cum_conf_1h', pd.Series(0, index=dataframe.index))

        # ── Exit LONG ────────────────────────────────────────────────
        # ML cumulative confidence drops below exit threshold (55%)
        dataframe.loc[
            (
                (ml_conf < self.exit_confidence_threshold) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1

        # ── Exit SHORT ───────────────────────────────────────────────
        # Same rule – confidence collapse means the ML no longer has conviction
        dataframe.loc[
            (
                (ml_conf < self.exit_confidence_threshold) &
                (dataframe['volume'] > 0)
            ),
            'exit_short'
        ] = 1

        n_exit_long = dataframe['exit_long'].sum() if 'exit_long' in dataframe.columns else 0
        n_exit_short = dataframe['exit_short'].sum() if 'exit_short' in dataframe.columns else 0
        logger.info(f"Exit signals – LONG: {n_exit_long}, SHORT: {n_exit_short}")

        return dataframe
