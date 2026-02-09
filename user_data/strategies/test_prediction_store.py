#!/usr/bin/env python3
"""
Test script for PredictionStore and cumulative confidence calculation.

Run from the freqtrade directory:
    python user_data/strategies/test_prediction_store.py
"""

import os
import sys
from datetime import datetime

# Add the strategies directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediction_store import PredictionStore


def test_cumulative_confidence():
    """Test the cumulative confidence algorithm with sample data."""
    
    print("=" * 60)
    print("Testing Cumulative Confidence Algorithm")
    print("=" * 60)
    
    # Use a temporary test database
    test_db = "/tmp/test_predictions.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    store = PredictionStore(
        db_path=test_db,
        ewma_alpha=0.3,
        consistency_window=4,
        boost_factor=1.1,
        penalty_factor=0.9
    )
    
    print(f"\nParameters:")
    print(f"  EWMA alpha: {store.ewma_alpha}")
    print(f"  Consistency window: {store.consistency_window}")
    print(f"  Boost factor: {store.boost_factor}")
    print(f"  Penalty factor: {store.penalty_factor}")
    
    # Test Case 1: Consistent Bullish Signals
    print("\n" + "-" * 60)
    print("Test Case 1: Consistent Bullish Signals (Building Confidence)")
    print("-" * 60)
    
    bullish_predictions = [
        {'direction': '看涨', 'confidence': 60, 'prediction_pct': 2.5, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.6, 'current_price': 90000},
        {'direction': '看涨', 'confidence': 65, 'prediction_pct': 2.8, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.65, 'current_price': 90100},
        {'direction': '看涨', 'confidence': 70, 'prediction_pct': 3.0, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.7, 'current_price': 90200},
        {'direction': '看涨', 'confidence': 75, 'prediction_pct': 3.2, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.75, 'current_price': 90300},
        {'direction': '看涨', 'confidence': 80, 'prediction_pct': 3.5, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.8, 'current_price': 90400},
    ]
    
    for i, pred in enumerate(bullish_predictions, 1):
        store.store_prediction(pred)
        latest = store.get_latest_prediction()
        print(f"  T{i}: direction={pred['direction']}, raw_conf={pred['confidence']}%, "
              f"cumulative={latest['cumulative_confidence']:.1f}%")
    
    final = store.get_latest_prediction()
    print(f"\n  Result: Cumulative confidence = {final['cumulative_confidence']:.1f}%")
    print(f"  Entry triggered: {final['cumulative_confidence'] >= 75} (threshold: 75%)")
    
    # Clear for next test
    os.remove(test_db)
    store = PredictionStore(db_path=test_db, ewma_alpha=0.3, consistency_window=4)
    
    # Test Case 2: Mixed Signals
    print("\n" + "-" * 60)
    print("Test Case 2: Mixed Signals (Suppressed Confidence)")
    print("-" * 60)
    
    mixed_predictions = [
        {'direction': '看涨', 'confidence': 70, 'prediction_pct': 2.5, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.7, 'current_price': 90000},
        {'direction': '看跌', 'confidence': 65, 'prediction_pct': -2.0, 'range': {'range_name': 'large_drop'}, 'signal_strength': 0.65, 'current_price': 89900},
        {'direction': '看涨', 'confidence': 72, 'prediction_pct': 2.2, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.72, 'current_price': 90000},
        {'direction': '震荡', 'confidence': 50, 'prediction_pct': 0.3, 'range': {'range_name': 'sideways'}, 'signal_strength': 0.5, 'current_price': 90050},
        {'direction': '看涨', 'confidence': 75, 'prediction_pct': 2.8, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.75, 'current_price': 90100},
    ]
    
    for i, pred in enumerate(mixed_predictions, 1):
        store.store_prediction(pred)
        latest = store.get_latest_prediction()
        print(f"  T{i}: direction={pred['direction']}, raw_conf={pred['confidence']}%, "
              f"cumulative={latest['cumulative_confidence']:.1f}%")
    
    final = store.get_latest_prediction()
    print(f"\n  Result: Cumulative confidence = {final['cumulative_confidence']:.1f}%")
    print(f"  Entry triggered: {final['cumulative_confidence'] >= 75} (threshold: 75%)")
    print(f"  Note: Despite high raw confidence (75%), mixed signals keep cumulative low")
    
    # Clear for next test
    os.remove(test_db)
    store = PredictionStore(db_path=test_db, ewma_alpha=0.3, consistency_window=4)
    
    # Test Case 3: Confidence Fading (Exit Trigger)
    print("\n" + "-" * 60)
    print("Test Case 3: Confidence Fading (Triggers Exit)")
    print("-" * 60)
    
    fading_predictions = [
        {'direction': '看涨', 'confidence': 85, 'prediction_pct': 3.5, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.85, 'current_price': 90000},
        {'direction': '看涨', 'confidence': 80, 'prediction_pct': 3.2, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.8, 'current_price': 90500},
        {'direction': '看涨', 'confidence': 70, 'prediction_pct': 2.8, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.7, 'current_price': 90800},
        {'direction': '看涨', 'confidence': 55, 'prediction_pct': 2.0, 'range': {'range_name': 'large_rise'}, 'signal_strength': 0.55, 'current_price': 91000},
        {'direction': '震荡', 'confidence': 45, 'prediction_pct': 0.5, 'range': {'range_name': 'sideways'}, 'signal_strength': 0.45, 'current_price': 91100},
    ]
    
    for i, pred in enumerate(fading_predictions, 1):
        store.store_prediction(pred)
        latest = store.get_latest_prediction()
        status = "HOLD" if latest['cumulative_confidence'] >= 75 else "EXIT!"
        print(f"  T{i}: direction={pred['direction']}, raw_conf={pred['confidence']}%, "
              f"cumulative={latest['cumulative_confidence']:.1f}% -> {status}")
    
    final = store.get_latest_prediction()
    print(f"\n  Result: Cumulative confidence dropped to {final['cumulative_confidence']:.1f}%")
    print(f"  Exit triggered: {final['cumulative_confidence'] < 75}")
    
    # Show statistics
    print("\n" + "-" * 60)
    print("Database Statistics")
    print("-" * 60)
    stats = store.get_statistics()
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Average confidence: {stats['avg_confidence']:.1f}%")
    print(f"  Direction distribution: {stats['direction_distribution']}")
    print(f"  Range distribution: {stats['range_distribution']}")
    
    # Cleanup
    os.remove(test_db)
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cumulative_confidence()
