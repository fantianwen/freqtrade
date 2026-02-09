"""
BTC Predictor Strategy for Freqtrade

Integrates with the BTC price predictor server to make trading decisions
based on ML model predictions and cumulative confidence.

Entry Conditions:
- |predicted_pct| > 2% (only trade "大涨" or "大跌")
- cumulative_confidence >= 75%

Exit Conditions:
- cumulative_confidence drops below 75%

Features:
- Polls predictor server every 15 seconds via bot_loop_start()
- Stores predictions in SQLite with cumulative confidence
- Supports both long and short positions
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, Trade
from freqtrade.persistence import Trade as TradeModel

# Import the prediction store
from user_data.strategies.prediction_store import PredictionStore

logger = logging.getLogger(__name__)


class BTCPredictorStrategy(IStrategy):
    """
    BTC Predictor Strategy - Uses ML predictions for entry/exit decisions.
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # Enable short trading for futures
    can_short: bool = True
    
    # Timeframe - we use 1m but poll predictions every 15s
    timeframe = "1m"
    
    # Minimal ROI - we rely on cumulative confidence for exits
    minimal_roi = {
        "0": 100  # Effectively disabled, use exit signals instead
    }
    
    # Stoploss - safety net
    stoploss = -0.05  # 5% stop loss as safety
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    # Use exit signal
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Process only new candles for indicators, but we poll more frequently
    process_only_new_candles = False
    
    # Startup candles needed
    startup_candle_count: int = 10
    
    # Order types
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }
    
    # ============ Predictor Configuration ============
    # These can be overridden in config.json under "predictor_config"
    
    # Predictor server URL
    predictor_url: str = "http://localhost:8080"
    
    # Polling interval in seconds
    poll_interval: int = 15
    
    # SQLite database path
    db_path: str = "user_data/predictions.db"
    
    # EWMA parameters
    ewma_alpha: float = 0.3
    consistency_window: int = 4
    
    # Trading thresholds
    entry_threshold_pct: float = 2.0  # Only trade when |prediction| > 2%
    confidence_threshold: float = 80.0  # Cumulative confidence entry/exit threshold
    min_confidence: float = 54.0  # Minimum raw confidence for entry
    
    # ============ Internal State ============
    _last_poll_time: Optional[datetime] = None
    _prediction_store: Optional[PredictionStore] = None
    _http_session: Optional[requests.Session] = None
    _latest_prediction: Optional[Dict] = None
    
    def __init__(self, config: dict) -> None:
        """Initialize the strategy."""
        super().__init__(config)
        
        # Load predictor config from freqtrade config
        predictor_config = config.get('predictor_config', {})
        
        self.predictor_url = predictor_config.get('server_url', self.predictor_url)
        self.poll_interval = predictor_config.get('poll_interval_seconds', self.poll_interval)
        self.db_path = predictor_config.get('db_path', self.db_path)
        self.ewma_alpha = predictor_config.get('ewma_alpha', self.ewma_alpha)
        self.consistency_window = predictor_config.get('consistency_window', self.consistency_window)
        self.entry_threshold_pct = predictor_config.get('entry_threshold_pct', self.entry_threshold_pct)
        self.confidence_threshold = predictor_config.get('confidence_threshold', self.confidence_threshold)
        self.min_confidence = predictor_config.get('min_confidence', self.min_confidence)
        
        logger.info(f"BTCPredictorStrategy initialized with predictor_url={self.predictor_url}")
        logger.info(f"Entry filters: min_confidence={self.min_confidence}, confidence_threshold={self.confidence_threshold}")
    
    def bot_start(self, **kwargs) -> None:
        """
        Called once at the start of the bot.
        Initialize prediction store and HTTP session.
        """
        logger.info("BTCPredictorStrategy bot_start() called")
        
        # Initialize prediction store
        self._prediction_store = PredictionStore(
            db_path=self.db_path,
            ewma_alpha=self.ewma_alpha,
            consistency_window=self.consistency_window
        )
        
        # Initialize HTTP session for API calls
        self._http_session = requests.Session()
        self._http_session.timeout = 10  # 10 second timeout
        
        # Initialize poll time
        self._last_poll_time = None
        
        logger.info(f"Prediction store initialized at {self.db_path}")
        logger.info(f"Will poll {self.predictor_url}/api/predict every {self.poll_interval}s")
    
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the beginning of each bot iteration.
        Polls the predictor server every poll_interval seconds.
        """
        # Check if it's time to poll
        if self._should_poll(current_time):
            self._poll_prediction()
            self._last_poll_time = current_time
    
    def _should_poll(self, current_time: datetime) -> bool:
        """Check if enough time has passed since last poll."""
        if self._last_poll_time is None:
            return True
        
        elapsed = (current_time - self._last_poll_time).total_seconds()
        return elapsed >= self.poll_interval
    
    def _poll_prediction(self) -> Optional[Dict]:
        """
        Poll the predictor server for a new prediction.
        
        Returns:
            Prediction dict or None if request failed
        """
        if self._http_session is None:
            logger.warning("HTTP session not initialized, skipping poll")
            return None
        
        try:
            response = self._http_session.get(
                f"{self.predictor_url}/api/predict",
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('success', False):
                logger.warning(f"Prediction API returned error: {data.get('error', 'Unknown')}")
                return None
            
            # Extract prediction data
            prediction = {
                'direction': data.get('direction', '震荡'),
                'range': data.get('range', {}),
                'confidence': data.get('confidence', 0),
                'signal_strength': data.get('signal_strength', 0),
                'prediction_pct': data.get('prediction_pct', 0),
                'current_price': data.get('current_price', 0),
            }
            
            # Store in database
            if self._prediction_store:
                self._prediction_store.store_prediction(prediction)
            
            # Cache latest prediction
            self._latest_prediction = prediction
            
            # Get updated cumulative confidence
            if self._prediction_store:
                latest = self._prediction_store.get_latest_prediction()
                if latest:
                    prediction['cumulative_confidence'] = latest['cumulative_confidence']
            
            logger.info(
                f"Prediction: {prediction['direction']} {prediction['prediction_pct']:.2f}%, "
                f"conf={prediction['confidence']:.1f}%, "
                f"cumulative={prediction.get('cumulative_confidence', 0):.1f}%"
            )
            
            return prediction
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to poll prediction server: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            return None
    
    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get the current prediction state for trading decisions.
        
        Returns:
            Dict with direction, prediction_pct, cumulative_confidence, etc.
        """
        if self._prediction_store is None:
            return {
                'direction': '震荡',
                'prediction_pct': 0,
                'cumulative_confidence': 0,
                'signal_type': 'neutral',
                'range': 'unknown'
            }
        
        latest = self._prediction_store.get_latest_prediction()
        
        if latest is None:
            return {
                'direction': '震荡',
                'prediction_pct': 0,
                'cumulative_confidence': 0,
                'signal_type': 'neutral',
                'range': 'unknown'
            }
        
        return {
            'direction': latest['direction'],
            'prediction_pct': latest['prediction_pct'],
            'cumulative_confidence': latest['cumulative_confidence'],
            'signal_type': latest['signal_type'],
            'range': latest['range'],
            'confidence': latest['confidence']
        }
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators - we add prediction-based columns.
        """
        # Get current prediction state
        state = self._get_current_state()
        
        # Add prediction data to dataframe (latest value for all rows)
        dataframe['pred_direction'] = state['direction']
        dataframe['pred_pct'] = state['prediction_pct']
        dataframe['pred_cumulative_conf'] = state['cumulative_confidence']
        dataframe['pred_signal_type'] = state['signal_type']
        dataframe['pred_range'] = state['range']
        
        # Add entry/exit condition flags
        is_large_move = abs(state['prediction_pct']) > self.entry_threshold_pct
        is_confident = state['cumulative_confidence'] >= self.confidence_threshold
        has_min_confidence = state.get('confidence', 0) >= self.min_confidence
        
        dataframe['pred_is_large_move'] = is_large_move
        dataframe['pred_is_confident'] = is_confident
        dataframe['pred_has_min_confidence'] = has_min_confidence
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals based on predictions.
        
        Entry conditions:
        - |prediction_pct| > 2% (大涨 or 大跌)
        - cumulative_confidence >= 75%
        """
        state = self._get_current_state()
        
        # Check entry conditions
        is_large_move = abs(state['prediction_pct']) > self.entry_threshold_pct
        is_confident = state['cumulative_confidence'] >= self.confidence_threshold
        has_min_confidence = state.get('confidence', 0) >= self.min_confidence
        
        # Long entry: 大涨 prediction with high confidence
        dataframe.loc[
            (
                (dataframe['pred_direction'] == '看涨') &
                (dataframe['pred_is_large_move'] == True) &
                (dataframe['pred_is_confident'] == True) &
                (dataframe['pred_has_min_confidence'] == True) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1
        
        # Short entry: 大跌 prediction with high confidence
        dataframe.loc[
            (
                (dataframe['pred_direction'] == '看跌') &
                (dataframe['pred_is_large_move'] == True) &
                (dataframe['pred_is_confident'] == True) &
                (dataframe['pred_has_min_confidence'] == True) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'
        ] = 1
        
        # Log entry signals
        if is_large_move and is_confident and has_min_confidence:
            logger.info(
                f"Entry signal: {state['direction']} with "
                f"pred={state['prediction_pct']:.2f}%, "
                f"conf={state.get('confidence', 0):.1f}%, "
                f"cumulative_conf={state['cumulative_confidence']:.1f}%"
            )
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals based on cumulative confidence.
        
        Exit condition:
        - cumulative_confidence drops below 75%
        """
        state = self._get_current_state()
        
        # Exit when confidence fades
        confidence_fading = state['cumulative_confidence'] < self.confidence_threshold
        
        # Exit long when confidence drops
        dataframe.loc[
            (
                (dataframe['pred_cumulative_conf'] < self.confidence_threshold) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'
        ] = 1
        
        # Exit short when confidence drops
        dataframe.loc[
            (
                (dataframe['pred_cumulative_conf'] < self.confidence_threshold) &
                (dataframe['volume'] > 0)
            ),
            'exit_short'
        ] = 1
        
        # Log exit signals
        if confidence_fading:
            logger.info(
                f"Exit signal: cumulative_conf={state['cumulative_confidence']:.1f}% "
                f"< threshold={self.confidence_threshold}%"
            )
        
        return dataframe
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        """
        Custom exit logic - exit when cumulative confidence drops.
        
        This provides more responsive exits than waiting for the next candle.
        """
        state = self._get_current_state()
        
        # Exit if cumulative confidence drops below threshold
        if state['cumulative_confidence'] < self.confidence_threshold:
            logger.info(
                f"Custom exit for {pair}: cumulative_conf={state['cumulative_confidence']:.1f}% "
                f"< {self.confidence_threshold}%, profit={current_profit:.2%}"
            )
            return f"confidence_fade_{state['cumulative_confidence']:.0f}"
        
        # Exit if direction reverses significantly
        if trade.is_short:
            # In short position, exit if prediction turns bullish
            if state['direction'] == '看涨' and state['cumulative_confidence'] >= 60:
                logger.info(f"Custom exit for {pair}: direction reversed to bullish")
                return "direction_reversal_bullish"
        else:
            # In long position, exit if prediction turns bearish
            if state['direction'] == '看跌' and state['cumulative_confidence'] >= 60:
                logger.info(f"Custom exit for {pair}: direction reversed to bearish")
                return "direction_reversal_bearish"
        
        return None
    
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Confirm trade entry - final check before placing order.
        """
        state = self._get_current_state()
        
        # Double-check conditions are still valid
        is_large_move = abs(state['prediction_pct']) > self.entry_threshold_pct
        is_confident = state['cumulative_confidence'] >= self.confidence_threshold
        has_min_confidence = state.get('confidence', 0) >= self.min_confidence
        
        if not (is_large_move and is_confident and has_min_confidence):
            logger.warning(
                f"Trade entry rejected for {pair}: conditions no longer met "
                f"(pred={state['prediction_pct']:.2f}%, conf={state.get('confidence', 0):.1f}%, "
                f"cumulative_conf={state['cumulative_confidence']:.1f}%)"
            )
            return False
        
        # Check direction matches side
        if side == 'long' and state['direction'] != '看涨':
            logger.warning(f"Trade entry rejected: long entry but direction is {state['direction']}")
            return False
        
        if side == 'short' and state['direction'] != '看跌':
            logger.warning(f"Trade entry rejected: short entry but direction is {state['direction']}")
            return False
        
        logger.info(
            f"Trade entry confirmed for {pair} ({side}): "
            f"pred={state['prediction_pct']:.2f}%, conf={state.get('confidence', 0):.1f}%, "
            f"cumulative_conf={state['cumulative_confidence']:.1f}%"
        )
        
        return True
    
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Determine leverage based on confidence level.
        
        Higher confidence = higher leverage (within limits)
        """
        state = self._get_current_state()
        confidence = state['cumulative_confidence']
        
        # Base leverage on confidence
        # 75% conf -> 2x, 85% conf -> 3x, 95% conf -> 4x
        if confidence >= 95:
            leverage = min(4.0, max_leverage)
        elif confidence >= 85:
            leverage = min(3.0, max_leverage)
        elif confidence >= 75:
            leverage = min(2.0, max_leverage)
        else:
            leverage = 1.0
        
        logger.info(f"Leverage for {pair}: {leverage}x (confidence={confidence:.1f}%)")
        
        return leverage
    
    def informative_pairs(self):
        """
        Define informative pairs - we don't need additional pairs
        as predictions come from the external server.
        """
        return []
