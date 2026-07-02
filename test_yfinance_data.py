#!/usr/bin/env python3
"""
Test script to validate NVIDIA stock data from YFinance.
Checks data quality, completeness, and consistency.
"""

import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

def load_and_validate_yfinance_data():
    """Load NVIDIA data from YFinance and run comprehensive validation."""
    
    print("=" * 80)
    print("YFinance NVIDIA (NVDA) Data Validation Test")
    print("=" * 80)
    
    # Configuration
    TICKER = "NVDA"
    START_DATE = "2015-01-01"
    END_DATE = "2024-12-31"
    
    # Attempt to load
    print(f"\n[1] Attempting to load {TICKER} from YFinance...")
    print(f"    Period: {START_DATE} to {END_DATE}")
    
    try:
        import yfinance as yf
        print(f"    ✓ yfinance version: {yf.__version__}")
    except ImportError:
        print("    ✗ yfinance not installed. Install with: pip install yfinance")
        return False
    
    try:
        raw = yf.download(TICKER, start=START_DATE, end=END_DATE, 
                         progress=True, auto_adjust=True)
        print(f"    ✓ Download successful")
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return False
    
    # Basic shape check
    print(f"\n[2] Data Shape & Basic Properties")
    print(f"    Shape: {raw.shape}")
    print(f"    Columns: {list(raw.columns)}")
    print(f"    Data types:\n{raw.dtypes}")
    
    # Check for MultiIndex columns (yfinance sometimes returns this)
    if isinstance(raw.columns, pd.MultiIndex):
        print(f"    ⚠ WARNING: MultiIndex columns detected, flattening...")
        raw.columns = raw.columns.get_level_values(0)
        print(f"    ✓ Flattened to: {list(raw.columns)}")
    
    # Extract required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in raw.columns]
    if missing_cols:
        print(f"    ✗ Missing columns: {missing_cols}")
        return False
    else:
        print(f"    ✓ All required columns present: {required_cols}")
    
    df = raw[required_cols].copy()
    
    # Date range validation
    print(f"\n[3] Date Range Validation")
    print(f"    First date: {df.index.min().date()}")
    print(f"    Last date:  {df.index.max().date()}")
    print(f"    Total trading days: {len(df):,}")
    
    expected_days = len(pd.bdate_range(START_DATE, END_DATE))
    if len(df) < expected_days * 0.95:
        print(f"    ⚠ WARNING: Only {len(df)} days vs expected ~{expected_days} (weekdays)")
    else:
        print(f"    ✓ Day count reasonable ({len(df)} ≈ {expected_days} expected)")
    
    # Missing values check
    print(f"\n[4] Missing Values Check")
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"    ✗ NaN values found:")
        print(nan_counts[nan_counts > 0])
        return False
    else:
        print(f"    ✓ No NaN values in any column")
    
    # Duplicates check
    print(f"\n[5] Duplicates Check")
    dup_dates = df.index.duplicated().sum()
    if dup_dates > 0:
        print(f"    ✗ {dup_dates} duplicate dates found")
        return False
    else:
        print(f"    ✓ No duplicate dates")
    
    # Price consistency checks
    print(f"\n[6] Price Consistency Checks")
    
    # High >= Low >= 0
    high_low_violation = (df["High"] < df["Low"]).sum()
    if high_low_violation > 0:
        print(f"    ✗ High < Low violations: {high_low_violation}")
        return False
    else:
        print(f"    ✓ High >= Low for all days")
    
    # Close within [Low, High]
    close_violation = ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum()
    if close_violation > 0:
        print(f"    ✗ Close outside [Low, High]: {close_violation} days")
        return False
    else:
        print(f"    ✓ Close is within [Low, High] for all days")
    
    # Open within [Low, High]
    open_violation = ((df["Open"] > df["High"]) | (df["Open"] < df["Low"])).sum()
    if open_violation > 0:
        print(f"    ✗ Open outside [Low, High]: {open_violation} days")
        return False
    else:
        print(f"    ✓ Open is within [Low, High] for all days")
    
    # Non-positive prices
    non_positive = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()
    if non_positive > 0:
        print(f"    ✗ Non-positive prices found: {non_positive} days")
        return False
    else:
        print(f"    ✓ All prices are positive")
    
    # Volume check
    print(f"\n[7] Volume Validation")
    zero_volume = (df["Volume"] == 0).sum()
    if zero_volume > 0:
        print(f"    ⚠ WARNING: {zero_volume} days with zero volume")
    else:
        print(f"    ✓ All days have non-zero volume")
    
    print(f"    Volume stats:")
    print(f"      Min: {df['Volume'].min():,.0f}")
    print(f"      Mean: {df['Volume'].mean():,.0f}")
    print(f"      Median: {df['Volume'].median():,.0f}")
    print(f"      Max: {df['Volume'].max():,.0f}")
    
    # Time gaps analysis
    print(f"\n[8] Time Gap Analysis (Calendar Days Between Trading Days)")
    gaps = df.index.to_series().diff().dt.days.dropna()
    gap_counts = gaps.value_counts().sort_index()
    print(f"    Gap distribution:")
    for gap, count in gap_counts.head(5).items():
        print(f"      {gap} day(s): {count} occurrences")
    
    long_gaps = gaps[gaps > 4]
    if len(long_gaps) > 0:
        print(f"    ⚠ Long gaps (>4 days): {len(long_gaps)} (likely holidays/trading halts)")
    
    # Price statistics
    print(f"\n[9] Price Statistics")
    print(f"    Close price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"    Close price mean: ${df['Close'].mean():.2f}")
    print(f"    Close price std: ${df['Close'].std():.2f}")
    print(f"    Price growth: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.1f}%")
    
    # Returns analysis
    print(f"\n[10] Returns Analysis")
    returns = df["Close"].pct_change().dropna()
    log_returns = np.log(df["Close"]).diff().dropna()
    
    print(f"    Daily return stats:")
    print(f"      Mean: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
    print(f"      Std: {returns.std():.6f} ({returns.std()*100:.4f}%)")
    print(f"      Min: {returns.min():.6f} ({returns.min()*100:.4f}%)")
    print(f"      Max: {returns.max():.6f} ({returns.max()*100:.4f}%)")
    print(f"      Skewness: {returns.skew():.4f}")
    print(f"      Kurtosis: {returns.kurtosis():.4f}")
    
    # Extreme return check
    extreme_returns = (returns.abs() > 0.20).sum()
    if extreme_returns > 0:
        print(f"    ⚠ Extreme single-day moves (>20%): {extreme_returns} days")
    
    # Final summary
    print(f"\n" + "=" * 80)
    print(f"VALIDATION RESULT: ✓ PASS - Data is suitable for analysis")
    print(f"=" * 80)
    
    # Save summary
    print(f"\n[11] Sample Data Preview (first 5 and last 5 rows):")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    
    return True


if __name__ == "__main__":
    success = load_and_validate_yfinance_data()
    sys.exit(0 if success else 1)
