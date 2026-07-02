#!/usr/bin/env python3
"""
Script to download and cache NVIDIA stock data from YFinance.
Saves data to CSV for offline fallback.
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

def get_data_directory():
    """Get or create data directory."""
    data_dir = Path("notebooks/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_cache_path():
    """Get path to cached CSV file."""
    return get_data_directory() / "nvda_ohlcv_cache.csv"

def download_yfinance_data(ticker="NVDA", start="2015-01-01", end="2024-12-31"):
    """
    Download stock data from YFinance.
    
    Args:
        ticker: Stock ticker (default: NVDA)
        start: Start date (default: 2015-01-01)
        end: End date (default: 2024-12-31)
    
    Returns:
        DataFrame or None if download failed
    """
    print(f"\n[DOWNLOAD] Attempting to fetch {ticker} from YFinance...")
    
    try:
        import yfinance as yf
        print(f"[DOWNLOAD] yfinance version: {yf.__version__}")
    except ImportError:
        print("[ERROR] yfinance not installed. Install with: pip install yfinance")
        return None
    
    try:
        raw = yf.download(ticker, start=start, end=end, 
                         progress=True, auto_adjust=True)
        print(f"[DOWNLOAD] ✓ Download successful - {len(raw)} rows")
        
        # Handle MultiIndex columns (yfinance quirk)
        if isinstance(raw.columns, pd.MultiIndex):
            print(f"[DOWNLOAD] ⚠ MultiIndex columns detected, flattening...")
            raw.columns = raw.columns.get_level_values(0)
        
        # Extract required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = raw[required_cols].copy()
        
        return df
    
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return None

def validate_data(df):
    """
    Validate data quality.
    
    Returns:
        (is_valid: bool, messages: list)
    """
    messages = []
    
    # Check for NaN
    if df.isna().sum().sum() > 0:
        messages.append(f"✗ NaN values found: {df.isna().sum().sum()}")
        return False, messages
    messages.append("✓ No NaN values")
    
    # Check duplicates
    if df.index.duplicated().sum() > 0:
        messages.append(f"✗ Duplicate dates: {df.index.duplicated().sum()}")
        return False, messages
    messages.append("✓ No duplicate dates")
    
    # Price consistency
    if (df["High"] < df["Low"]).sum() > 0:
        messages.append(f"✗ High < Low violations: {(df['High'] < df['Low']).sum()}")
        return False, messages
    messages.append("✓ High >= Low for all days")
    
    if ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum() > 0:
        messages.append("✗ Close outside [Low, High]")
        return False, messages
    messages.append("✓ Close is within [Low, High]")
    
    if (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum() > 0:
        messages.append("✗ Non-positive prices found")
        return False, messages
    messages.append("✓ All prices are positive")
    
    # Volume check
    if (df["Volume"] == 0).sum() > 0:
        messages.append(f"⚠ {(df['Volume'] == 0).sum()} days with zero volume")
    else:
        messages.append("✓ All days have non-zero volume")
    
    return True, messages

def save_to_cache(df):
    """Save data to CSV cache."""
    cache_path = get_cache_path()
    try:
        df.to_csv(cache_path)
        print(f"[CACHE] ✓ Data saved to: {cache_path}")
        print(f"[CACHE]   File size: {cache_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save cache: {e}")
        return False

def load_from_cache():
    """Load data from CSV cache."""
    cache_path = get_cache_path()
    
    if not cache_path.exists():
        print(f"[CACHE] Cache file not found: {cache_path}")
        return None
    
    try:
        print(f"[CACHE] Loading from cache: {cache_path}")
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        print(f"[CACHE] ✓ Loaded {len(df)} rows from cache")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load cache: {e}")
        return None

def get_nvda_data(ticker="NVDA", start="2015-01-01", end="2024-12-31", 
                  force_download=False):
    """
    Get NVIDIA data with fallback strategy.
    
    Strategy:
    1. Try to download from YFinance (if not force_download=False)
    2. Validate downloaded data
    3. Save to cache
    4. If download fails, load from cache
    5. If cache missing, use synthetic data
    
    Args:
        ticker: Stock ticker
        start: Start date
        end: End date
        force_download: If True, skip cache and always try to download
    
    Returns:
        (df, source): DataFrame and source name
    """
    
    print("\n" + "="*80)
    print("NVIDIA Stock Data Loader")
    print("="*80)
    
    # Try download first (unless explicitly skipped)
    if not force_download:
        print("\n[STEP 1] Attempting YFinance download...")
        df = download_yfinance_data(ticker, start, end)
        
        if df is not None:
            print("\n[STEP 2] Validating downloaded data...")
            is_valid, messages = validate_data(df)
            for msg in messages:
                print(f"[VALIDATE] {msg}")
            
            if is_valid:
                print("\n[STEP 3] Saving to cache...")
                save_to_cache(df)
                print("\n[SUCCESS] ✓ Using live YFinance data")
                return df, "yfinance"
    
    # Fallback to cache
    print("\n[STEP 2] Attempting to load from cache...")
    df = load_from_cache()
    
    if df is not None:
        print("\n[VALIDATE] Cache data validation:")
        is_valid, messages = validate_data(df)
        for msg in messages:
            print(f"[VALIDATE] {msg}")
        
        if is_valid:
            print("\n[SUCCESS] ✓ Using cached data")
            return df, "cache"
        else:
            print("[WARNING] Cache data failed validation")
    
    # Final fallback: synthetic data
    print("\n[FALLBACK] Using synthetic data (placeholder)")
    print("[WARNING] This is for testing only - use real data for analysis!")
    return None, "none"

def print_data_summary(df):
    """Print summary statistics of loaded data."""
    if df is None:
        return
    
    print("\n" + "="*80)
    print("Data Summary")
    print("="*80)
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Trading days: {len(df):,}")
    print(f"\nClose price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"Close price mean: ${df['Close'].mean():.2f}")
    print(f"Volume (mean): {df['Volume'].mean():,.0f}")
    print(f"Price growth: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.1f}%")
    
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nLast 3 rows:")
    print(df.tail(3))

if __name__ == "__main__":
    # Test the data loader
    df, source = get_nvda_data()
    
    if df is not None:
        print_data_summary(df)
        print(f"\n[INFO] Data source: {source}")
    else:
        print("\n[ERROR] Failed to load any data")
        sys.exit(1)
