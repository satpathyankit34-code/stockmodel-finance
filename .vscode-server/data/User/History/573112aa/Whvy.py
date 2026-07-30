import os
import random
import shutil
import numpy as np
import pandas as pd
import yfinance as yf

# Force Matplotlib to non-interactive backend BEFORE importing pyplot/mplfinance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Clean directory name optimized for PyTorch (ImageFolder) and TensorFlow
OUTPUT_DIR = "stock_dataset"
CLASSES = ["buy", "sell", "hold", "unknown"]
MAX_PER_CLASS = 1000      # Target limit of images per class total

# Data Window Settings
WINDOW_SIZE = 40          # Days shown in candlestick chart
FUTURE_LOOKAHEAD = 5      # Days ahead to compute target return
STRIDE = 1                # Step by 1 day to maximize image count per ticker

# Thresholds tuned for balanced class distribution
BUY_THRESHOLD = 0.020     # > 2.0% gain -> buy
SELL_THRESHOLD = -0.020   # < -2.0% loss -> sell
HOLD_LIMIT = 0.005        # Within [-0.5%, 0.5%] -> hold

# Expanded S&P 500 Ticker List (120+ tickers)
TICKERS = [
    # Tech / Semis / Software
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AVGO", "CSCO", "ADBE",
    "CRM", "INTC", "AMD", "QCOM", "ORCL", "TXN", "AMAT", "NOW", "INTU", "MU",
    "LRCX", "PANW", "SNPS", "CDNS", "KLAC", "IBM", "AMCR", "ADI", "FI", "FIS",
    # Finance / Banking
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP",
    "SCHW", "PGR", "CB", "MMC", "AIG", "MET", "BK", "COF", "USB", "PNC",
    # Healthcare / Pharma / Biotech
    "JNJ", "UNH", "PFE", "ABT", "MRK", "TMO", "LLY", "DHR", "BMY", "AMGN",
    "GILD", "CVS", "CI", "ISRG", "SYK", "MDT", "REGN", "VRTX", "BDX", "ZTS",
    # Consumer / Retail / Automotive
    "WMT", "PG", "HD", "KO", "PEP", "COST", "MCD", "NKE", "PM", "SBUX",
    "TGT", "LOW", "EL", "CL", "KMB", "MDLZ", "DG", "DLTR", "ORLY", "AZO",
    # Industrials / Energy / Aerospace
    "XOM", "CVX", "GE", "CAT", "BA", "HON", "UPS", "UNP", "RTX", "LMT",
    "DE", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "FDX", "NSC", "GD"
]

# Track global counts per category
class_counts = {
    "train": {c: 0 for c in CLASSES},
    "val": {c: 0 for c in CLASSES}
}

def is_class_full(cls):
    return (class_counts["train"][cls] + class_counts["val"][cls]) >= MAX_PER_CLASS

def all_classes_full():
    return all(is_class_full(cls) for cls in CLASSES)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def setup_directories():
    """
    Creates a standard ML directory structure compatible with 
    PyTorch's ImageFolder and TensorFlow's image_dataset_from_directory:
    
    stock_dataset/
    ├── train/
    │   ├── buy/
    │   ├── sell/
    │   ├── hold/
    │   └── unknown/
    └── val/
        ├── buy/
        ├── sell/
        ├── hold/
        └── unknown/
    """
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing old dataset at '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
        
    print(f"Initializing clean training directory structure at '{OUTPUT_DIR}'...")
    for split in ["train", "val"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

def generate_chart_image(df, save_path):
    """
    Renders monochrome candlestick chart:
    - Up days: Hollow body
    - Down days: Solid white body
    - Zero margin, zero axes, dark background (#121212)
    """
    mc = mpf.make_marketcolors(
        up='none',
        down='white',
        edge='white',
        wick='white'
    )
    s = mpf.make_mpf_style(marketcolors=mc, figcolor='#121212', facecolor='#121212')
    
    fig, axlist = mpf.plot(
        df,
        type='candle',
        style=s,
        volume=False,
        returnfig=True,
        figratio=(4, 3),
        figscale=1.0,
        axisoff=True
    )
    
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='#121212', dpi=96)
    plt.close(fig)

# ==========================================
# DATA PROCESSING & LABELING
# ==========================================
def process_ticker(ticker_symbol, split_folder):
    if all_classes_full():
        return

    print(f"Processing {ticker_symbol} [{split_folder.upper()}]...")
    
    try:
        df = yf.download(ticker_symbol, period="5y", interval="1d", progress=False)
    except Exception as e:
        print(f"  -> Download failed for {ticker_symbol}: {e}")
        return

    if df.empty or len(df) < (WINDOW_SIZE + FUTURE_LOOKAHEAD + 10):
        print(f"  -> Skipping {ticker_symbol} (Insufficient Data)")
        return

    # Flatten MultiIndex columns from recent yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove potential duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna()
    
    local_count = 0

    # Extract raw numpy price values to avoid Series/DataFrame scalar issues
    close_prices = df['Close'].values

    # Sliding window with STRIDE = 1
    for i in range(WINDOW_SIZE, len(df) - FUTURE_LOOKAHEAD, STRIDE):
        if all_classes_full():
            break

        window_df = df.iloc[i - WINDOW_SIZE : i]
        
        current_price = float(close_prices[i - 1])
        future_price = float(close_prices[i + FUTURE_LOOKAHEAD - 1])
        percent_change = (future_price - current_price) / current_price
        
        if percent_change >= BUY_THRESHOLD:
            label = "buy"
        elif percent_change <= SELL_THRESHOLD:
            label = "sell"
        elif abs(percent_change) <= HOLD_LIMIT:
            label = "hold"
        else:
            label = "unknown"
            
        if is_class_full(label):
            continue
            
        filename = f"{ticker_symbol}_{df.index[i].strftime('%Y%m%d')}.png"
        save_path = os.path.join(OUTPUT_DIR, split_folder, label, filename)
        
        try:
            generate_chart_image(window_df, save_path)
            class_counts[split_folder][label] += 1
            local_count += 1
        except Exception as err:
            if local_count == 0 and i == WINDOW_SIZE:
                print(f"  -> Rendering error on first window: {err}")
            continue
            
    print(f"  -> Generated {local_count} images for {ticker_symbol}")

def print_summary():
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE - FINAL IMAGE COUNTS")
    print("="*50)
    for split in ["train", "val"]:
        print(f"\n--- [{split.upper()} SPLIT] ---")
        total = 0
        for cls in CLASSES:
            cnt = class_counts[split][cls]
            total += cnt
            print(f"  {cls.upper():<8}: {cnt:,} images")
        print(f"  TOTAL   : {total:,} images")
    print("="*50)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    setup_directories()
    
    # Enforce Ticker-Disjoint Split (80% Train / 20% Val)
    random.seed(42)
    tickers_shuffled = TICKERS.copy()
    random.shuffle(tickers_shuffled)
    
    split_index = int(len(tickers_shuffled) * 0.8)
    train_tickers = tickers_shuffled[:split_index]
    val_tickers = tickers_shuffled[split_index:]
    
    print(f"Split Strategy: {len(train_tickers)} Tickers TRAIN | {len(val_tickers)} Tickers VAL")
    print("-" * 50)
    
    # Process Train Split
    for ticker in train_tickers:
        if all_classes_full():
            break
        process_ticker(ticker, "train")
        
    # Process Validation Split
    for ticker in val_tickers:
        if all_classes_full():
            break
        process_ticker(ticker, "val")
        
    print_summary()
    print("\nDirectories are now set up at './stock_dataset/' and ready for training!")