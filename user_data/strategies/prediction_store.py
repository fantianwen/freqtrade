"""
Prediction Store - SQLite storage for BTC price predictions

Handles:
- Storing predictions from the predictor server
- Calculating cumulative confidence using EWMA with direction consistency
- Querying recent predictions for trading decisions
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PredictionStore:
    """
    SQLite-based storage for BTC price predictions with cumulative confidence calculation.
    """
    
    def __init__(
        self,
        db_path: str = "user_data/predictions.db",
        ewma_alpha: float = 0.3,
        consistency_window: int = 4,
        boost_factor: float = 1.1,
        penalty_factor: float = 0.9
    ):
        """
        Initialize the prediction store.
        
        Args:
            db_path: Path to SQLite database file
            ewma_alpha: EWMA decay factor (0.3 = 30% weight to current, 70% to history)
            consistency_window: Number of recent predictions to check for direction consistency
            boost_factor: Multiplier for consistent direction (default 1.1 = 10% boost)
            penalty_factor: Multiplier for mixed signals (default 0.9 = 10% penalty)
        """
        self.db_path = db_path
        self.ewma_alpha = ewma_alpha
        self.consistency_window = consistency_window
        self.boost_factor = boost_factor
        self.penalty_factor = penalty_factor
        
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT NOT NULL,
                range TEXT NOT NULL,
                prediction_time DATETIME NOT NULL,
                confidence REAL NOT NULL,
                signal_strength REAL NOT NULL,
                signal_type TEXT NOT NULL,
                cumulative_confidence REAL NOT NULL,
                prediction_pct REAL NOT NULL,
                current_price REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_time 
            ON predictions(prediction_time DESC)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Prediction store initialized at {self.db_path}")
    
    def calculate_cumulative_confidence(
        self,
        current_confidence: float,
        current_direction: str,
        recent_predictions: Optional[List[Dict]] = None
    ) -> float:
        """
        Calculate cumulative confidence using EWMA with direction consistency.
        
        Algorithm:
        1. Apply Exponential Weighted Moving Average to smooth confidence
        2. Check direction consistency in recent window
        3. Apply boost for consistent direction, penalty for mixed signals
        
        Args:
            current_confidence: Current prediction confidence (0-100)
            current_direction: Current direction ("看涨", "看跌", "震荡")
            recent_predictions: List of recent predictions (if None, fetches from DB)
            
        Returns:
            Cumulative confidence value (0-100)
        """
        if recent_predictions is None:
            recent_predictions = self.get_recent_predictions(limit=self.consistency_window)
        
        # If no history, return current confidence
        if not recent_predictions:
            return current_confidence
        
        # Calculate EWMA
        # Start with oldest prediction and work forward
        confidences = [p['confidence'] for p in reversed(recent_predictions)]
        confidences.append(current_confidence)
        
        ewma = confidences[0]
        for conf in confidences[1:]:
            ewma = self.ewma_alpha * conf + (1 - self.ewma_alpha) * ewma
        
        # Check direction consistency (including current prediction)
        recent_directions = [p['direction'] for p in recent_predictions[-self.consistency_window:]]
        recent_directions.append(current_direction)
        
        # Only consider last consistency_window directions
        check_directions = recent_directions[-self.consistency_window:]
        unique_directions = set(check_directions)
        
        # Apply boost/penalty based on consistency
        if len(unique_directions) == 1 and current_direction != '震荡':
            # All same non-neutral direction: apply boost
            ewma = min(100, ewma * self.boost_factor)
            logger.debug(f"Direction consistent ({current_direction}), applying {self.boost_factor}x boost")
        elif len(unique_directions) > 2 or '震荡' in unique_directions:
            # Mixed or contains neutral: apply penalty
            ewma = ewma * self.penalty_factor
            logger.debug(f"Mixed signals detected, applying {self.penalty_factor}x penalty")
        
        return round(ewma, 2)
    
    def store_prediction(self, prediction: Dict[str, Any]) -> int:
        """
        Store a prediction from the predictor server.
        
        Args:
            prediction: Prediction dict from the API with keys:
                - direction: "看涨", "看跌", "震荡"
                - range: range dict with 'range_name'
                - confidence: 0-100
                - signal_strength: multi-timeframe signal strength
                - prediction_pct: predicted price change %
                - current_price: current BTC price
                
        Returns:
            ID of the inserted record
        """
        # Determine signal_type from direction
        direction = prediction.get('direction', '震荡')
        if direction == '看涨':
            signal_type = 'bullish'
        elif direction == '看跌':
            signal_type = 'bearish'
        else:
            signal_type = 'neutral'
        
        # Get range name
        range_info = prediction.get('range', {})
        range_name = range_info.get('range_name', 'unknown') if isinstance(range_info, dict) else str(range_info)
        
        # Calculate cumulative confidence
        cumulative_confidence = self.calculate_cumulative_confidence(
            current_confidence=prediction.get('confidence', 0),
            current_direction=direction
        )
        
        # Insert into database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (
                direction, range, prediction_time, confidence, 
                signal_strength, signal_type, cumulative_confidence,
                prediction_pct, current_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            direction,
            range_name,
            datetime.now().isoformat(),
            prediction.get('confidence', 0),
            prediction.get('signal_strength', 0),
            signal_type,
            cumulative_confidence,
            prediction.get('prediction_pct', 0),
            prediction.get('current_price', 0)
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(
            f"Stored prediction #{record_id}: {direction} {range_name}, "
            f"conf={prediction.get('confidence', 0):.1f}%, "
            f"cumulative={cumulative_confidence:.1f}%"
        )
        
        return record_id
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent predictions ordered by time (newest first).
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dicts
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM predictions 
            ORDER BY prediction_time DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_latest_prediction(self) -> Optional[Dict]:
        """
        Get the most recent prediction.
        
        Returns:
            Latest prediction dict or None if no predictions exist
        """
        predictions = self.get_recent_predictions(limit=1)
        return predictions[0] if predictions else None
    
    def get_current_cumulative_confidence(self) -> float:
        """
        Get the current cumulative confidence.
        
        Returns:
            Current cumulative confidence or 0 if no predictions
        """
        latest = self.get_latest_prediction()
        return latest['cumulative_confidence'] if latest else 0.0
    
    def get_predictions_since(self, since: datetime) -> List[Dict]:
        """
        Get all predictions since a specific time.
        
        Args:
            since: Datetime to query from
            
        Returns:
            List of predictions since that time
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE prediction_time >= ?
            ORDER BY prediction_time DESC
        """, (since.isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def cleanup_old_predictions(self, days: int = 7):
        """
        Remove predictions older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM predictions 
            WHERE prediction_time < ?
        """, (cutoff.isoformat(),))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old predictions")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about stored predictions.
        
        Returns:
            Dict with count, avg confidence, direction distribution, etc.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) as count FROM predictions")
        total = cursor.fetchone()['count']
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
        avg_conf = cursor.fetchone()['avg_conf'] or 0
        
        # Direction distribution
        cursor.execute("""
            SELECT direction, COUNT(*) as count 
            FROM predictions 
            GROUP BY direction
        """)
        direction_dist = {row['direction']: row['count'] for row in cursor.fetchall()}
        
        # Range distribution
        cursor.execute("""
            SELECT range, COUNT(*) as count 
            FROM predictions 
            GROUP BY range
        """)
        range_dist = {row['range']: row['count'] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_predictions': total,
            'avg_confidence': round(avg_conf, 2),
            'direction_distribution': direction_dist,
            'range_distribution': range_dist
        }
