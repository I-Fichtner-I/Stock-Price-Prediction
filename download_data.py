#!/usr/bin/env python3
"""
Script to download and cache NVIDIA stock data from YFinance.
Saves data to CSV for offline fallback.
Always loads data up to TODAY.
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, date
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

def download_yfinance_data(ticker="NVDA", start="2015-01-01", end=None):
    """
    Download stock data from YFinance up to today.
    
    Args:
        ticker: Stock ticker (default: NVDA)
        start: Start date (default: 2015-01-01)
        end: End date (default: today). If None, uses today's date.
    
    Returns:
        DataFrame or None if download failed
    """
    if end is None:
        end = date.today().isoformat()
    
    print(f"\n[DOWNLOAD] Attempting to fetch {ticker} from YFinance...")
    print(f"[DOWNLOAD] Period: {start} to {end} (TODAY)")
    
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
        print(f"[DOWNLOAD] Data range: {raw.index.min().date()} to {raw.index.max().date()}")
        
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
        file_size_mb = cache_path.stat().st_size / (1024*1024)
        print(f"[CACHE] ✓ Data saved to: {cache_path}")
        print(f"[CACHE]   File size: {file_size_mb:.2f} MB")
        print(f"[CACHE]   Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        print(f"[CACHE] Data range: {df.index.min().date()} to {df.index.max().date()}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load cache: {e}")
        return None

def get_cache_age():
    """Get age of cache file in days."""
    cache_path = get_cache_path()
    
    if not cache_path.exists():
        return None
    
    mtime = cache_path.stat().st_mtime
    cache_date = datetime.fromtimestamp(mtime)
    age = (datetime.now() - cache_date).days
    return age

def get_nvda_data(ticker="NVDA", start="2015-01-01", end=None,
                  force_download=False, refresh_if_older_than_days=7):
    """
    Get NVIDIA data with fallback strategy, always loading up to TODAY.
    
    Strategy:
    1. Check cache age - if too old, force refresh
    2. Try to download from YFinance (if not force_download=False)
    3. Validate downloaded data
    4. Save to cache
    5. If download fails, load from cache
    6. If cache missing, use synthetic data
    
    Args:
        ticker: Stock ticker
        start: Start date
        end: End date (if None, uses today)
        force_download: If True, skip cache and always try to download
        refresh_if_older_than_days: Refresh cache if older than N days
    
    Returns:
        (df, source, cache_age): DataFrame, source name, cache age in days
    """
    
    if end is None:
        end = date.today().isoformat()
    
    print("\n" + "="*80)
    print(f"NVIDIA Stock Data Loader - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check cache age
    cache_age = get_cache_age()
    if cache_age is not None:
        print(f"\n[INFO] Cache age: {cache_age} days")
        if cache_age >= refresh_if_older_than_days:
            print(f"[INFO] Cache is older than {refresh_if_older_than_days} days - will refresh")
            force_download = True
    
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
                print("\n[SUCCESS] ✓ Using live YFinance data (up to TODAY)")
                return df, "yfinance", 0
    
    # Fallback to cache
    print("\n[STEP 2] Attempting to load from cache...")
    df = load_from_cache()
    
    if df is not None:
        print("\n[VALIDATE] Cache data validation:")
        is_valid, messages = validate_data(df)
        for msg in messages:
            print(f"[VALIDATE] {msg}")
        
        if is_valid:
            print("\n[WARNING] Using cached data (may be older than today)")
            age = get_cache_age()
            if age is not None and age > 0:
                print(f"[WARNING] Cache is {age} day(s) old")
            print("[SUCCESS] ✓ Using cached data")
            return df, "cache", age if age is not None else -1
        else:
            print("[WARNING] Cache data failed validation")
    
    # Final fallback: synthetic data
    print("\n[FALLBACK] Using synthetic data (placeholder)")
    print("[WARNING] This is for testing only - use real data for analysis!")
    return None, "none", -1

def print_data_summary(df):
    """Print summary statistics of loaded data."""
    if df is None:
        print("[ERROR] No data to summarize")
        return
    
    print("\n" + "="*80)
    print("Data Summary")
    print("="*80)
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Trading days: {len(df):,}")
    print(f"\nClose price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"Close price mean: ${df['Close'].mean():.2f}")
    print(f"Close price current (latest): ${df['Close'].iloc[-1]:.2f}")
    print(f"Volume (mean): {df['Volume'].mean():,.0f}")
    print(f"Price growth (total): {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.1f}%")
    
    # Recent returns
    recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
    print(f"Return (last 5 trading days): {recent_return:.2f}%")
    
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nLast 3 rows:")
    print(df.tail(3))

if __name__ == "__main__":
    # Test the data loader with today's date
    df, source, cache_age = get_nvda_data()
    
    if df is not None:
        print_data_summary(df)
        print(f"\n[INFO] Data source: {source}")
        if cache_age >= 0:
            print(f"[INFO] Cache age: {cache_age} days")
    else:
        print("\n[ERROR] Failed to load any data")
        sys.exit(1)
