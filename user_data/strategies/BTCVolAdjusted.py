"""
BTC Volatility-Adjusted Directional Strategy
==============================================

Combines 4 ML models for intelligent trading:
  1. Direction model    – predicts price change % over ~20h
  2. Vol regression     – predicts realized volatility magnitude (2h)
  3. Range regression   – predicts price range % (2h)  -> SL/TP sizing
  4. Vol classifier     – predicts high/low volatility (2h) -> position sizing

Entry:  Direction > 0.5% + VZO consensus + Range > 0.5%
Exit:   Direction flip, consecutive sideways, or VZO divergence
Size:   Halved in high-vol, boosted at 4h boundaries

Usage:
    freqtrade download-data --config user_data/config_btc_vol_adjusted.json \
        --timeframes 5m 15m 30m 1h 4h --timerange 20240101- --trading-mode futures

    freqtrade backtesting --strategy BTCVolAdjusted \
        --config user_data/config_btc_vol_adjusted.json \
        --timerange 20240601-20260201 --export trades
"""

import logging
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair

logger = logging.getLogger(__name__)


# =====================================================================
# Technical Indicators  (identical to BTCVZOBacktest / collect_and_train)
# =====================================================================

class TechnicalIndicators:
    @staticmethod
    def calculate_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_kdj(df, n=9, m1=3, m2=3):
        low_n = df['low'].rolling(n).min()
        high_n = df['high'].rolling(n).max()
        rsv = ((df['close'] - low_n) / (high_n - low_n) * 100).fillna(50)
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3*k - 2*d
        return k, d, j

    @staticmethod
    def calculate_macd(close, fast=12, slow=26, signal=9):
        ef = TechnicalIndicators.calculate_ema(close, fast)
        es = TechnicalIndicators.calculate_ema(close, slow)
        macd = ef - es
        sig = TechnicalIndicators.calculate_ema(macd, signal)
        return macd, sig, macd - sig

    @staticmethod
    def detect_crossover(fast, slow):
        golden = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        death  = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        return golden, death

    @staticmethod
    def calculate_rsi(close, period=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        ag = gain.ewm(alpha=1/period, adjust=False).mean()
        al = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = ag / al
        return (100 - 100/(1+rs)).fillna(50).replace([np.inf, -np.inf], 50)

    @staticmethod
    def calculate_roc(close, period=10):
        return ((close - close.shift(period)) / close.shift(period) * 100).fillna(0)

    @staticmethod
    def calculate_momentum(close, period=10):
        return (close - close.shift(period)).fillna(0)

    @staticmethod
    def calculate_williams_r(df, period=14):
        hn = df['high'].rolling(period).max()
        ln = df['low'].rolling(period).min()
        return ((hn - df['close']) / (hn - ln) * -100).fillna(-50)

    @staticmethod
    def calculate_cci(df, period=20):
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return ((tp - sma) / (0.015 * mad)).fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def calculate_adx(df, period=14):
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        up = h - h.shift(1)
        dn = l.shift(1) - l
        pdm = up.where((up > dn) & (up > 0), 0)
        mdm = dn.where((dn > up) & (dn > 0), 0)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        pdi = 100 * pdm.ewm(alpha=1/period, adjust=False).mean() / atr
        mdi = 100 * mdm.ewm(alpha=1/period, adjust=False).mean() / atr
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return (adx.fillna(25).replace([np.inf,-np.inf],25),
                pdi.fillna(25).replace([np.inf,-np.inf],25),
                mdi.fillna(25).replace([np.inf,-np.inf],25))

    @staticmethod
    def calculate_stoch_rsi(close, rsi_p=14, stoch_p=14):
        rsi = TechnicalIndicators.calculate_rsi(close, rsi_p)
        rl = rsi.rolling(stoch_p).min()
        rh = rsi.rolling(stoch_p).max()
        k = ((rsi - rl) / (rh - rl) * 100).fillna(50)
        d = k.rolling(3).mean().fillna(50)
        return k, d


# =====================================================================
# VZO helpers
# =====================================================================

def calculate_vzo(df, period=14, ma_len=9):
    sv = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
    vp = pd.Series(sv, index=df.index).ewm(span=period, adjust=False).mean()
    tv = df['volume'].ewm(span=period, adjust=False).mean()
    vzo = (100 * vp / tv).fillna(0).replace([np.inf, -np.inf], 0)
    vzo_ma = vzo.ewm(span=ma_len, adjust=False).mean()
    return vzo, vzo_ma


def calculate_vzo_slope(vzo, lookback=5):
    x = np.arange(lookback, dtype=float)
    xm = x.mean()
    xv = ((x - xm)**2).sum()

    def _lr(w):
        if len(w) < lookback:
            return np.nan
        y = w.values
        return ((x - xm) * (y - y.mean())).sum() / xv

    return vzo.rolling(lookback).apply(_lr, raw=False).fillna(0)


# =====================================================================
# ██  Strategy
# =====================================================================

class BTCVolAdjusted(IStrategy):
    """
    Volatility-adjusted directional strategy.
    Base timeframe: 1h (aligned with model training).
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '1h'

    # ROI disabled -- rely on exit signals + custom stoploss
    minimal_roi = {"0": 100}

    # Hard floor stoploss -- custom_stoploss overrides with range-based SL
    # At 5x leverage, -0.25 allows 5% price movement (same as -0.05 at 1x)
    stoploss = -0.25
    use_custom_stoploss = True

    # No trailing -- we use explicit exit logic
    trailing_stop = False

    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True

    startup_candle_count = 200

    # ── Internal ──
    _ti = TechnicalIndicators()
    _models: Dict[str, dict] = {}   # model_key -> {model, scaler, feature_names}

    # ================================================================
    # Init
    # ================================================================

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._models = {}
        pc = config.get('predictor_config', {})

        # Load all 4 models
        model_specs = {
            'direction': {
                'path': pc.get('direction_model_path', ''),
                'model_key': 'best_model',
            },
            'vol_regression': {
                'path': pc.get('vol_regression_model_path', ''),
                'model_key': 'model',
            },
            'range_regression': {
                'path': pc.get('range_regression_model_path', ''),
                'model_key': 'model',
            },
            'vol_classifier': {
                'path': pc.get('vol_classifier_model_path', ''),
                'model_key': 'model',
            },
        }

        for name, spec in model_specs.items():
            path = spec['path']
            if not path:
                logger.warning(f"Model '{name}': no path configured")
                continue

            # Resolve relative paths against the config file directory or userdir
            if not os.path.isabs(path):
                # Try several base directories
                candidates = [
                    path,                                               # as-is (cwd)
                    os.path.join(config.get('user_data_dir', ''), '..', path),  # relative to userdir parent
                ]
                # Also try relative to common freqtrade roots
                for base in [os.getcwd(), os.path.dirname(os.path.abspath(__file__)), '/home/ubuntu/alex-trading']:
                    candidates.append(os.path.join(base, path))
                    candidates.append(os.path.join(base, '..', path))

                resolved = None
                for c in candidates:
                    c = os.path.normpath(c)
                    if os.path.exists(c):
                        resolved = c
                        break
                if resolved:
                    path = resolved
                else:
                    logger.warning(f"Model '{name}' not found. Tried: {[os.path.normpath(c) for c in candidates[:4]]}")
                    continue
            elif not os.path.exists(path):
                logger.warning(f"Model '{name}' not found at: {path}")
                continue
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self._models[name] = {
                    'model': data.get(spec['model_key']),
                    'scaler': data.get('scaler'),
                    'feature_names': data.get('feature_names', []),
                }
                n_feat = len(self._models[name]['feature_names'])
                logger.info(f"Loaded {name}: {type(self._models[name]['model']).__name__}, "
                            f"{n_feat} features")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")

        logger.info(f"Models loaded: {list(self._models.keys())}")

    # ================================================================
    # Informative pairs
    # ================================================================

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, tf) for pair in pairs for tf in ['5m', '15m', '30m', '4h']]

    # ================================================================
    # Feature extraction (per timeframe)
    # ================================================================

    def _extract_tf_features(self, df: DataFrame, tf: str) -> dict:
        """Compute all features for one timeframe. Returns dict of Series."""
        ti = self._ti
        k, d, j = ti.calculate_kdj(df)
        macd, sig, hist = ti.calculate_macd(df['close'])
        rsi7 = ti.calculate_rsi(df['close'], 7)
        rsi14 = ti.calculate_rsi(df['close'], 14)
        rsi21 = ti.calculate_rsi(df['close'], 21)
        kg, kd = ti.detect_crossover(k, d)
        mg, md = ti.detect_crossover(macd, sig)

        ret = df['close'].pct_change()
        pm = {'5m':288*365,'15m':96*365,'30m':48*365,'1h':24*365,'4h':6*365}
        ann = np.sqrt(pm.get(tf, 24*365))
        vol_ann = ret.rolling(24).std() * ann

        v = df['volume']
        vma5, vma10, vma20 = v.rolling(5).mean(), v.rolling(10).mean(), v.rolling(20).mean()
        vr5 = (v/vma5).fillna(1).replace([np.inf,-np.inf],1)
        vr10 = (v/vma10).fillna(1).replace([np.inf,-np.inf],1)
        vr20 = (v/vma20).fillna(1).replace([np.inf,-np.inf],1)
        vc1 = v.pct_change(1).fillna(0).replace([np.inf,-np.inf],0)*100
        vc5 = v.pct_change(5).fillna(0).replace([np.inf,-np.inf],0)*100
        vt = ((vma5-vma20)/vma20*100).fillna(0).replace([np.inf,-np.inf],0)
        vh20, vl20 = v.rolling(20).max(), v.rolling(20).min()
        vpos = ((v-vl20)/(vh20-vl20)).fillna(0.5)
        vspk = (vr20 > 2).astype(int)
        vshr = (vr20 < 0.5).astype(int)
        pu = (df['close'] > df['close'].shift(1)).astype(int)
        vu = (v > v.shift(1)).astype(int)
        vpd = (pu != vu).astype(int)

        rh = df['high'].rolling(20).max()
        rl = df['low'].rolling(20).min()
        ppos = ((df['close']-rl)/(rh-rl)).fillna(0.5)
        ma20 = df['close'].rolling(20).mean()
        ts = ((df['close']-ma20)/ma20*100).fillna(0).replace([np.inf,-np.inf],0)

        rob = (rsi14 > 70).astype(int)
        ros = (rsi14 < 30).astype(int)
        rtr = (rsi14 - rsi14.shift(5)).fillna(0)

        roc5 = ti.calculate_roc(df['close'],5)
        roc10 = ti.calculate_roc(df['close'],10)
        roc20 = ti.calculate_roc(df['close'],20)
        mom10 = ti.calculate_momentum(df['close'],10)
        mom20 = ti.calculate_momentum(df['close'],20)
        wr = ti.calculate_williams_r(df,14)
        cci = ti.calculate_cci(df,20)
        adx, pdi, mdi = ti.calculate_adx(df,14)
        srk, srd = ti.calculate_stoch_rsi(df['close'])

        ccob = (cci>100).astype(int)
        ccos = (cci<-100).astype(int)
        adxs = (adx>25).astype(int)
        adxw = (adx<20).astype(int)
        trbull = (pdi>mdi).astype(int)

        # VZO
        vzo, vzo_ma = calculate_vzo(df)
        slope = calculate_vzo_slope(vzo, 5)
        saccel = slope.diff().fillna(0)
        vrm = vzo.rolling(20).mean().bfill()
        vrs = vzo.rolling(20).std().fillna(1).replace(0,1)
        vz = ((vzo-vrm)/vrs).fillna(0).replace([np.inf,-np.inf],0)
        srm = slope.rolling(20).mean().bfill()
        srs = slope.rolling(20).std().fillna(1).replace(0,1)
        sz = ((slope-srm)/srs).fillna(0).replace([np.inf,-np.inf],0)

        vzn = pd.Series(0, index=df.index)
        vzn = vzn.where(~(vzo>40), 2)
        vzn = vzn.where(~((vzo>15)&(vzo<=40)), 1)
        vzn = vzn.where(~((vzo>=-40)&(vzo<-15)), -1)
        vzn = vzn.where(~(vzo<-40), -2)

        ssn = pd.Series(0, index=df.index)
        ssn = ssn.where(~(slope>0), 1)
        ssn = ssn.where(~(slope<0), -1)

        p = tf
        return {
            f'{p}_kdj_k':k, f'{p}_kdj_d':d, f'{p}_kdj_j':j,
            f'{p}_kdj_golden':kg.astype(int).fillna(0), f'{p}_kdj_death':kd.astype(int).fillna(0),
            f'{p}_macd':macd, f'{p}_macd_signal':sig, f'{p}_macd_hist':hist,
            f'{p}_macd_golden':mg.astype(int).fillna(0), f'{p}_macd_death':md.astype(int).fillna(0),
            f'{p}_volatility':vol_ann.fillna(0),
            f'{p}_vol_ratio_ma5':vr5, f'{p}_vol_ratio_ma10':vr10, f'{p}_vol_ratio_ma20':vr20,
            f'{p}_vol_change_1':vc1, f'{p}_vol_change_5':vc5, f'{p}_vol_trend':vt,
            f'{p}_vol_position':vpos, f'{p}_vol_spike':vspk, f'{p}_vol_shrink':vshr,
            f'{p}_vol_price_divergence':vpd,
            f'{p}_price_position':ppos, f'{p}_trend_strength':ts,
            f'{p}_rsi_7':rsi7, f'{p}_rsi_14':rsi14, f'{p}_rsi_21':rsi21,
            f'{p}_rsi_overbought':rob, f'{p}_rsi_oversold':ros, f'{p}_rsi_trend':rtr,
            f'{p}_roc_5':roc5, f'{p}_roc_10':roc10, f'{p}_roc_20':roc20,
            f'{p}_mom_10':mom10, f'{p}_mom_20':mom20, f'{p}_williams_r':wr,
            f'{p}_cci':cci, f'{p}_cci_overbought':ccob, f'{p}_cci_oversold':ccos,
            f'{p}_adx':adx, f'{p}_plus_di':pdi, f'{p}_minus_di':mdi,
            f'{p}_adx_strong_trend':adxs, f'{p}_adx_weak_trend':adxw, f'{p}_trend_bullish':trbull,
            f'{p}_stoch_rsi_k':srk, f'{p}_stoch_rsi_d':srd,
            f'{p}_vzo':vzo, f'{p}_vzo_ma':vzo_ma, f'{p}_vzo_slope':slope,
            f'{p}_vzo_zscore':vz, f'{p}_slope_zscore':sz, f'{p}_slope_accel':saccel,
            f'{p}_vzo_zone':vzn, f'{p}_slope_sign':ssn,
        }

    # ================================================================
    # Vol-specific features (from collect_and_train_vol.py)
    # ================================================================

    def _add_vol_specific_features(self, feat: dict, df_1h: DataFrame,
                                   df_dict: Dict[str, DataFrame]) -> dict:
        """Add features useful for volatility prediction."""
        close = df_1h['close']
        high = df_1h['high']
        low = df_1h['low']
        returns = close.pct_change()

        for w in [6, 12, 24, 48]:
            feat[f'recent_vol_{w}h'] = returns.rolling(w).std().fillna(0)

        vol24 = returns.rolling(24).std()
        feat['vol_of_vol'] = vol24.rolling(12).std().fillna(0)
        vol6 = returns.rolling(6).std().fillna(0)
        feat['vol_ratio_6_24'] = (vol6 / vol24.replace(0, 1e-8)).fillna(1).replace([np.inf,-np.inf],1)

        for period in [7, 14, 20]:
            tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().fillna(0)
            feat[f'atr_{period}_pct'] = (atr/close*100).fillna(0).replace([np.inf,-np.inf],0)

        atr7 = (high-low).rolling(7).mean()
        atr20 = (high-low).rolling(20).mean()
        feat['atr_expansion'] = (atr7/atr20.replace(0,1e-8)).fillna(1).replace([np.inf,-np.inf],1)

        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bbw = (2*std20/ma20.replace(0,1e-8)*100).fillna(0).replace([np.inf,-np.inf],0)
        feat['bb_width_pct'] = bbw
        bbm = bbw.rolling(48).mean().bfill()
        bbs = bbw.rolling(48).std().fillna(1).replace(0,1)
        feat['bb_width_zscore'] = ((bbw-bbm)/bbs).fillna(0).replace([np.inf,-np.inf],0)

        loghl = np.log(high / low.replace(0,1e-8))
        park = np.sqrt(loghl**2 / (4*np.log(2)))
        feat['parkinson_vol_12'] = park.rolling(12).mean().fillna(0)

        body = (close - df_1h['open']).abs()
        wick = (high - low).replace(0, 1e-8)
        br = (body / wick).fillna(0.5)
        feat['body_ratio'] = br
        feat['body_ratio_ma5'] = br.rolling(5).mean().fillna(0.5)

        # 5m realized vol mapped to 1h
        if '5m' in df_dict and len(df_dict['5m']) > 50:
            df5 = df_dict['5m'].copy()
            df5['date'] = pd.to_datetime(df5['date'])
            ret5 = df5['close'].pct_change()
            for w, label in [(12,'1h'),(24,'2h'),(72,'6h')]:
                rv = ret5.rolling(w).std().fillna(0)
                rv_df = pd.DataFrame({'date': df5['date'], f'rv5m_{label}': rv})
                base_ts = pd.DataFrame({'date': pd.to_datetime(df_1h['date'])}, index=df_1h.index)
                merged = pd.merge_asof(base_ts.sort_values('date'), rv_df.sort_values('date'),
                                       on='date', direction='backward')
                merged.index = df_1h.index
                feat[f'rv5m_{label}'] = merged[f'rv5m_{label}'].fillna(0)

            if 'rv5m_1h' in feat and 'rv5m_2h' in feat:
                feat['rv5m_trend'] = (feat['rv5m_1h'] / feat['rv5m_2h'].replace(0,1e-8)).fillna(1)

        return feat

    # ================================================================
    # Build full feature matrix + run all models
    # ================================================================

    def _build_features_and_predict(self, df_1h: DataFrame,
                                     df_dict: Dict[str, DataFrame]) -> DataFrame:
        """
        Build features for every 1h candle, run all 4 models.
        Returns df_1h with prediction columns added.
        """
        ALL_TFS = ['5m', '15m', '30m', '1h', '4h']

        # Base features from 1h
        all_feat = self._extract_tf_features(df_1h, '1h')

        # Merge other timeframes via merge_asof
        for tf in ALL_TFS:
            if tf == '1h':
                continue
            if tf not in df_dict or len(df_dict[tf]) < 50:
                continue
            tf_df = df_dict[tf].copy()
            tf_feat = self._extract_tf_features(tf_df, tf)

            tf_feat_df = pd.DataFrame(tf_feat, index=tf_df.index)
            tf_feat_df['date'] = pd.to_datetime(tf_df['date'])
            base = pd.DataFrame({'date': pd.to_datetime(df_1h['date'])}, index=df_1h.index)

            merged = pd.merge_asof(
                base.sort_values('date'),
                tf_feat_df.sort_values('date'),
                on='date', direction='backward',
            )
            merged.index = df_1h.index
            for col in tf_feat:
                all_feat[col] = merged[col].fillna(0).replace([np.inf,-np.inf],0)

        # Cross-TF VZO consensus
        avail = [t for t in ALL_TFS if f'{t}_vzo_zone' in all_feat]
        if len(avail) >= 2:
            zone_df = pd.DataFrame({t: all_feat[f'{t}_vzo_zone'] for t in avail}, index=df_1h.index)
            sign_df = pd.DataFrame({t: all_feat[f'{t}_slope_sign'] for t in avail}, index=df_1h.index)
            all_feat['vzo_multi_tf_bullish'] = (zone_df > 0).sum(axis=1)
            all_feat['vzo_multi_tf_bearish'] = (zone_df < 0).sum(axis=1)
            all_feat['vzo_multi_tf_consensus'] = all_feat['vzo_multi_tf_bullish'] - all_feat['vzo_multi_tf_bearish']
            all_feat['slope_multi_tf_bullish'] = (sign_df > 0).sum(axis=1)
            all_feat['slope_multi_tf_bearish'] = (sign_df < 0).sum(axis=1)
            all_feat['slope_multi_tf_consensus'] = all_feat['slope_multi_tf_bullish'] - all_feat['slope_multi_tf_bearish']

            stfs = [t for t in ['5m','15m'] if f'{t}_vzo' in all_feat]
            ltfs = [t for t in ['1h','4h'] if f'{t}_vzo' in all_feat]
            if stfs and ltfs:
                all_feat['vzo_short_long_diff'] = (
                    pd.DataFrame({t:all_feat[f'{t}_vzo'] for t in stfs}, index=df_1h.index).mean(axis=1) -
                    pd.DataFrame({t:all_feat[f'{t}_vzo'] for t in ltfs}, index=df_1h.index).mean(axis=1))
                all_feat['slope_short_long_diff'] = (
                    pd.DataFrame({t:all_feat[f'{t}_vzo_slope'] for t in stfs}, index=df_1h.index).mean(axis=1) -
                    pd.DataFrame({t:all_feat[f'{t}_vzo_slope'] for t in ltfs}, index=df_1h.index).mean(axis=1))

        # Cross signals
        gc = [f'{t}_kdj_golden' for t in ALL_TFS if f'{t}_kdj_golden' in all_feat]
        gc += [f'{t}_macd_golden' for t in ALL_TFS if f'{t}_macd_golden' in all_feat]
        dc = [f'{t}_kdj_death' for t in ALL_TFS if f'{t}_kdj_death' in all_feat]
        dc += [f'{t}_macd_death' for t in ALL_TFS if f'{t}_macd_death' in all_feat]

        feat_df = pd.DataFrame(all_feat, index=df_1h.index)
        feat_df['multi_tf_golden_count'] = feat_df[gc].sum(axis=1) if gc else 0
        feat_df['multi_tf_death_count'] = feat_df[dc].sum(axis=1) if dc else 0
        feat_df['signal_strength'] = feat_df['multi_tf_golden_count'] - feat_df['multi_tf_death_count']

        # Time features
        dt = pd.to_datetime(df_1h['date'])
        feat_df['hour'] = dt.dt.hour
        feat_df['day_of_week'] = dt.dt.weekday
        feat_df['is_weekend'] = (dt.dt.weekday >= 5).astype(int)

        # Vol-specific features (needs dict form temporarily)
        feat_dict = {col: feat_df[col] for col in feat_df.columns}
        feat_dict['base_timestamp'] = df_1h['date']
        feat_dict = self._add_vol_specific_features(feat_dict, df_1h, df_dict)
        for col in feat_dict:
            if col not in feat_df.columns and col != 'base_timestamp':
                feat_df[col] = feat_dict[col]

        feat_df = feat_df.fillna(0).replace([np.inf, -np.inf], 0)

        # ── Run all models ──
        for model_name in ['direction', 'vol_regression', 'range_regression', 'vol_classifier']:
            if model_name not in self._models:
                continue
            m = self._models[model_name]
            try:
                X = feat_df.reindex(columns=m['feature_names'], fill_value=0)
                X = X.fillna(0).replace([np.inf, -np.inf], 0)
                if m['scaler'] is not None:
                    X_sc = pd.DataFrame(
                        m['scaler'].transform(X),
                        columns=m['feature_names'], index=X.index,
                    )
                else:
                    X_sc = X
                preds = m['model'].predict(X_sc)
                df_1h[f'ml_{model_name}'] = preds
            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {e}")
                df_1h[f'ml_{model_name}'] = 0

        # Vol classifier probability
        if 'vol_classifier' in self._models:
            try:
                m = self._models['vol_classifier']
                X = feat_df.reindex(columns=m['feature_names'], fill_value=0)
                X = X.fillna(0).replace([np.inf, -np.inf], 0)
                if m['scaler'] is not None:
                    X_sc = pd.DataFrame(
                        m['scaler'].transform(X),
                        columns=m['feature_names'], index=X.index,
                    )
                else:
                    X_sc = X
                proba = m['model'].predict_proba(X_sc)
                # Probability of the predicted class
                pred_cls = df_1h['ml_vol_classifier'].values.astype(int)
                df_1h['ml_vol_prob'] = [proba[i, c] for i, c in enumerate(pred_cls)]
            except Exception as e:
                logger.warning(f"Vol classifier proba failed: {e}")
                df_1h['ml_vol_prob'] = 0.5

        # Derived columns
        pred_pct = df_1h.get('ml_direction', pd.Series(0, index=df_1h.index))
        df_1h['ml_pred_pct'] = pred_pct
        df_1h['ml_dir_label'] = 'Sideways'
        df_1h.loc[pred_pct > 0.5, 'ml_dir_label'] = 'Bullish'
        df_1h.loc[pred_pct < -0.5, 'ml_dir_label'] = 'Bearish'

        df_1h['ml_range_pct'] = df_1h.get('ml_range_regression', pd.Series(0, index=df_1h.index))
        df_1h['ml_vol_class'] = df_1h.get('ml_vol_classifier', pd.Series(0, index=df_1h.index)).astype(int)

        # VZO consensus
        df_1h['ml_vzo_consensus'] = feat_df.get('vzo_multi_tf_consensus',
                                                  pd.Series(0, index=df_1h.index))

        # 4h boundary flag
        df_1h['ml_is_4h'] = (pd.to_datetime(df_1h['date']).dt.hour % 4 == 0).astype(int)

        # Previous predictions (for consecutive sideways exit)
        df_1h['ml_pred_pct_prev1'] = df_1h['ml_pred_pct'].shift(1).fillna(0)
        df_1h['ml_pred_pct_prev2'] = df_1h['ml_pred_pct'].shift(2).fillna(0)

        return df_1h

    # ================================================================
    # populate_indicators
    # ================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        logger.info(f"BTCVolAdjusted: populating indicators for {pair}, "
                    f"{len(dataframe)} candles @ {self.timeframe}")

        # Collect informative data
        df_dict: Dict[str, DataFrame] = {}
        if self.dp:
            for tf in ['5m', '15m', '30m', '4h']:
                inf = self.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                if len(inf) > 0:
                    df_dict[tf] = inf.copy()

        # The base 1h dataframe IS our main dataframe
        dataframe = self._build_features_and_predict(dataframe, df_dict)

        # Fill NaN
        pred_cols = [c for c in dataframe.columns if c.startswith('ml_')]
        for c in pred_cols:
            dataframe[c] = dataframe[c].fillna(0)

        logger.info(f"Indicators done. Direction model loaded: {'direction' in self._models}, "
                    f"Vol models: {[k for k in self._models if k != 'direction']}")
        return dataframe

    # ================================================================
    # Entry signals
    # ================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Guard: skip signals if models are not loaded
        if not self._models:
            logger.warning("No ML models loaded -- skipping entry signals")
            return dataframe

        pred = dataframe['ml_pred_pct']
        rng = dataframe['ml_range_pct']
        vzo_c = dataframe['ml_vzo_consensus']

        # LONG: strong direction + meaningful range + strong VZO alignment
        dataframe.loc[
            (pred > 1.0) &
            (rng > 1.0) &
            (vzo_c >= 3) &
            (dataframe['volume'] > 0),
            'enter_long'
        ] = 1

        # SHORT: strong direction + meaningful range + strong VZO alignment
        dataframe.loc[
            (pred < -1.0) &
            (rng > 1.0) &
            (vzo_c <= -3) &
            (dataframe['volume'] > 0),
            'enter_short'
        ] = 1

        nl = dataframe.get('enter_long', pd.Series(0)).sum()
        ns = dataframe.get('enter_short', pd.Series(0)).sum()
        logger.info(f"Entry signals: LONG={nl}, SHORT={ns}")
        return dataframe

    # ================================================================
    # Exit signals
    # ================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Guard: skip signals if models are not loaded
        if not self._models:
            logger.warning("No ML models loaded -- skipping exit signals")
            return dataframe

        pred = dataframe['ml_pred_pct']
        prev1 = dataframe['ml_pred_pct_prev1']
        prev2 = dataframe['ml_pred_pct_prev2']

        # Sideways = |pred| < 1.0 (matches entry threshold)
        sideways_now = pred.abs() < 1.0
        sideways_prev1 = prev1.abs() < 1.0
        sideways_prev2 = prev2.abs() < 1.0
        three_sideways = sideways_now & sideways_prev1 & sideways_prev2

        # Exit LONG when direction flips strongly bearish
        # OR 3 consecutive sideways candles (conviction lost)
        dataframe.loc[
            (
                (pred < -1.0) |
                three_sideways
            ) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1

        # Exit SHORT when direction flips strongly bullish
        # OR 3 consecutive sideways candles
        dataframe.loc[
            (
                (pred > 1.0) |
                three_sideways
            ) &
            (dataframe['volume'] > 0),
            'exit_short'
        ] = 1

        nel = dataframe.get('exit_long', pd.Series(0)).sum()
        nes = dataframe.get('exit_short', pd.Series(0)).sum()
        logger.info(f"Exit signals: LONG={nel}, SHORT={nes}")
        return dataframe

    # ================================================================
    # Custom stoploss (range-based)
    # ================================================================

    def custom_stoploss(self, pair: str, trade: Trade,
                        current_time, current_rate, current_profit,
                        after_fill, **kwargs) -> Optional[float]:
        """
        Dynamic stoploss based on predicted 2h price range.
        Leverage-aware: scale SL so that the allowed *price* movement
        equals the full predicted range regardless of leverage.

        At 5x leverage, 1% price move = 5% profit change,
        so SL = -(range_pct / 100) * leverage.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return self.stoploss

        last = dataframe.iloc[-1]
        range_pct = float(last.get('ml_range_pct', 1.0))
        lev = trade.leverage or 1.0

        # Allow full predicted range as price movement
        # Convert to profit-ratio by multiplying by leverage
        sl = -(range_pct / 100.0) * lev

        # Clamp: floor at hard stoploss, ceiling so at least 1% price move is allowed
        sl = max(sl, self.stoploss)           # no wider than hard stoploss
        sl = min(sl, -0.01 * lev)             # no tighter than 1% price move
        return sl

    # ================================================================
    # Leverage
    # ================================================================

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: Optional[str], side: str, **kwargs) -> float:
        """Fixed 5x leverage for all trades."""
        return 5.0

    # ================================================================
    # Custom stake amount (volatility-adjusted position sizing)
    # ================================================================

    def custom_stake_amount(self, pair: str, current_time,
                            current_rate: float, proposed_stake: float,
                            min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        """
        Adjust position size based on:
        - Vol regime (half in high-vol)
        - 4h boundary (1.25x boost)
        - Direction confidence (scale by prediction magnitude)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return proposed_stake

        last = dataframe.iloc[-1]

        base = proposed_stake

        # Half size in high-vol regime
        vol_class = int(last.get('ml_vol_class', 0))
        if vol_class == 1:
            base *= 0.5

        # Boost at 4h boundary
        is_4h = int(last.get('ml_is_4h', 0))
        if is_4h == 1:
            base *= 1.25

        # Scale by prediction confidence (how far from 0)
        pred_pct = abs(float(last.get('ml_pred_pct', 0)))
        confidence_factor = min(pred_pct / 2.0, 1.0)
        base *= max(confidence_factor, 0.3)  # floor at 30% of base

        # Clamp
        base = max(base, min_stake or 0)
        base = min(base, max_stake)

        return base
