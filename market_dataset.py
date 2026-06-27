"""
Market Dataset Pipeline
=======================
Loads USDJPY 1-minute candle data from the EA log and prepares
sliding windows for frame encoding and research experiments.

Features engineered per window:
  - Price returns (normalized, removes absolute price level)
  - Spread (liquidity signal)
  - BB position (price relative to bands)
  - MA slope (trend direction)
  - Candle body/range ratio (bar structure)
  - Window tick count variability (activity proxy)

Usage:
    from market_dataset import MarketDataset
    ds = MarketDataset("data/usdjpy_1m.csv")
    windows, labels = ds.get_windows(window=30, horizon=10)
"""

import numpy as np
import pandas as pd
from pathlib import Path


class MarketDataset:
    def __init__(self, path: str):
        self.path = Path(path)
        self.df   = self._load()

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        mid = (df["bid"] + df["ask"]) / 2

        # Returns (log, normalized within window later)
        df["return"]     = mid.pct_change()

        # Spread in pips (USDJPY: 1 pip = 0.01, spread col is in points *100)
        df["spread_pips"] = df["spread"] / 100.0

        # BB position: where is close relative to bands? -1 to +1
        bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_pos"] = ((df["close"] - df["bb_lower"]) / bb_range) * 2 - 1

        # MA slope (normalized by price)
        df["ma_slope"] = df["ma10"].diff() / mid

        # Candle structure: body / total range
        body  = (df["close"] - df["open"]).abs()
        range_ = (df["high"] - df["low"]).replace(0, np.nan)
        df["body_ratio"] = body / range_

        # Direction of candle
        df["direction"] = np.sign(df["close"] - df["open"])

        df = df.dropna().reset_index(drop=True)
        print(f"  Loaded {len(df):,} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        return df

    def get_windows(self, window: int = 30, horizon: int = 10):
        """
        Create sliding windows of `window` bars each.
        Label = sign of forward return over `horizon` bars (1=up, -1=down, 0=flat).
        Returns (windows: np.ndarray [N, window, features], labels: np.ndarray [N])
        """
        features = ["return", "spread_pips", "bb_pos", "ma_slope", "body_ratio", "direction"]
        X = self.df[features].values.astype(np.float32)

        mid    = ((self.df["bid"] + self.df["ask"]) / 2).values
        n      = len(X)
        windows, labels = [], []

        for i in range(window, n - horizon):
            w = X[i - window: i].copy()

            # Normalize returns within window (zero-mean, unit std)
            ret = w[:, 0]
            std = ret.std()
            if std > 1e-8:
                w[:, 0] = (ret - ret.mean()) / std

            # Forward return label
            fwd = (mid[i + horizon] - mid[i]) / mid[i]
            if fwd > 0.0002:    label = 1
            elif fwd < -0.0002: label = -1
            else:               label = 0

            windows.append(w)
            labels.append(label)

        windows = np.array(windows, dtype=np.float32)
        labels  = np.array(labels,  dtype=np.int8)

        up   = (labels ==  1).sum()
        down = (labels == -1).sum()
        flat = (labels ==  0).sum()
        print(f"  Windows: {len(windows):,}  (up={up:,} down={down:,} flat={flat:,})")
        print(f"  Window shape: {windows.shape}  →  [N, {window} bars, {len(features)} features]")
        return windows, labels

    def get_text_descriptions(self, n_samples: int = 500, window: int = 30) -> list:
        """
        Convert market windows to text for frame encoding experiments.
        Useful for testing Coda's semantic understanding of market structure.
        """
        features = ["return", "spread_pips", "bb_pos", "ma_slope", "body_ratio", "direction"]
        X   = self.df[features].values.astype(np.float32)
        mid = ((self.df["bid"] + self.df["ask"]) / 2).values
        n   = len(X)

        descriptions = []
        indices = np.random.choice(range(window, n - 10), size=min(n_samples, n - window - 10), replace=False)

        for i in indices:
            w        = X[i - window: i]
            trend    = "uptrend" if w[:, 5].mean() > 0.2 else ("downtrend" if w[:, 5].mean() < -0.2 else "ranging")
            vol      = "high volatility" if w[:, 0].std() > w[:, 0].std() * 1.5 else "normal volatility"
            spread   = "wide spread" if w[:, 1].mean() > 0.5 else "tight spread"
            bb       = "overbought" if w[-1, 2] > 0.7 else ("oversold" if w[-1, 2] < -0.7 else "mid-range")
            ma_dir   = "rising MA" if w[-1, 3] > 0 else "falling MA"
            desc = f"USDJPY {window}-bar window: {trend}, {vol}, {spread}, {bb}, {ma_dir}"
            descriptions.append(desc)

        return descriptions


if __name__ == "__main__":
    ds = MarketDataset("data/usdjpy_1m.csv")
    windows, labels = ds.get_windows(window=30, horizon=10)

    print(f"\nSample window [0]:")
    print(f"  Shape : {windows[0].shape}")
    print(f"  Return mean : {windows[0][:,0].mean():.4f}")
    print(f"  BB pos range: {windows[0][:,2].min():.2f} to {windows[0][:,2].max():.2f}")

    print(f"\nSample text descriptions:")
    descs = ds.get_text_descriptions(n_samples=5)
    for d in descs:
        print(f"  {d}")
