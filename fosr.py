import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

# --- CONFIGURATION ---
# Split tickers to prevent the model from memorizing specific companies (Ticker-Disjoint Split)
TRAIN_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'SPY', 'QQQ', 'AMD', 'NFLX', 'DIS', 'INTC']
VAL_TICKERS = ['BA', 'PYPL', 'ADBE', 'CRM', 'NFLX', 'PEP', 'COST'] # Unseen validation companies

OUTPUT_DIR = 'data/stock_dataset_v2'
CLASSES = ['buy', 'hold', 'sell', 'unknown']

TARGET_IMAGES_PER_CLASS = 2000
WINDOW_SIZE = 30

# Ensure directory structure exists
def setup_directories():
    for split in ['train', 'val']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# --- 1. VOLATILITY-ADJUSTED LABELING & NORMALIZATION ---
def fetch_and_generate_candlesticks():
    print("Fetching market data and rendering volatility-normalized charts...")
    
    datasets = {
        'train': {cls: [] for cls in ['buy', 'hold', 'sell']},
        'val': {cls: [] for cls in ['buy', 'hold', 'sell']}
    }
    counts = {'train': {cls: 0 for cls in ['buy', 'hold', 'sell']},
              'val': {cls: 0 for cls in ['buy', 'hold', 'sell']}}

    all_tickers = [(t, 'train') for t in TRAIN_TICKERS] + [(t, 'val') for t in VAL_TICKERS]

    for ticker, split in all_tickers:
        try:
            df = yf.download(ticker, period='max', interval='1d', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            if len(df) < WINDOW_SIZE + 5:
                continue

            for i in range(len(df) - WINDOW_SIZE - 5):
                window = df.iloc[i:i+WINDOW_SIZE].copy()
                
                # Dynamic Volatility Thresholding using Average True Range (ATR) approximation
                high_low = window['High'] - window['Low']
                high_close = np.abs(window['High'] - window['Shift'] if 'Shift' in window else window['Close'].shift())
                true_range = np.maximum(high_low, high_close)
                atr_threshold = true_range.mean() / window.iloc[-1]['Close'] * 1.5 # Adaptive threshold
                
                future_window = df.iloc[i+WINDOW_SIZE : i+WINDOW_SIZE+5]
                future_max = future_window['High'].max()
                future_min = future_window['Low'].min()
                current_price = window.iloc[-1]['Close']

                # Triple-barrier style directional check
                max_gain = (future_max - current_price) / current_price
                max_loss = (future_min - current_price) / current_price

                if max_gain > max(0.02, atr_threshold) and max_gain > abs(max_loss):
                    label = 'buy'
                elif max_loss < -max(0.02, atr_threshold) and abs(max_loss) > max_gain:
                    label = 'sell'
                else:
                    label = 'hold'

                if counts[split][label] >= TARGET_IMAGES_PER_CLASS:
                    continue

                # Min-Max Normalize price series to 0 - 100 scale to fix scale distortion
                for col in ['Open', 'High', 'Low', 'Close']:
                    min_val = window[col].min()
                    max_val = window[col].max()
                    if max_val - min_val > 0:
                        window[col] = (window[col] - min_val) / (max_val - min_val) * 100.0
                    else:
                        window[col] = 50.0

                datasets[split][label].append((window, ticker, i))
                counts[split][label] += 1

        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")

    # Render out to disk
    for split in ['train', 'val']:
        for label, items in datasets[split].items():
            print(f"Rendering {len(items)} images for [{split.upper()} -> {label}]")
            for idx, (window, ticker, window_id) in enumerate(tqdm(items, desc=f"{split}-{label}")):
                filename = f"{ticker}_{window_id}.png"
                filepath = os.path.join(OUTPUT_DIR, split, label, filename)

                style = random.choice(['binance', 'yahoo', 'charles'])
                try:
                    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
                    s = mpf.make_mpf_style(marketcolors=mc, base_style=style)

                    mpf.plot(
                        window, 
                        type='candle', 
                        style=s, 
                        axisoff=True, 
                        savefig=dict(fname=filepath, dpi=100, bbox_inches='tight', pad_inches=0.1)
                    )
                except Exception:
                    continue

# --- 2. ROBUST UNKNOWN CLASS GENERATOR ---
def generate_unknowns():
    print("Generating robust unknown/distractor images...")
    total_structured = TARGET_IMAGES_PER_CLASS * 3
    
    for split, ratio in [('train', 0.8), ('val', 0.2)]:
        unknown_target = int(total_structured * ratio)
        for i in tqdm(range(unknown_target), desc=f"{split}-unknown"):
            filepath = os.path.join(OUTPUT_DIR, split, 'unknown', f"distractor_{i}.png")

            # Complex noise/patterns to force the network to learn structural rejection
            img = Image.new('RGB', (300, 300), color=(random.randint(20, 240), random.randint(20, 240), random.randint(20, 240)))
            draw = ImageDraw.Draw(img)

            noise_type = random.choice(['bars', 'circles', 'waves', 'noise_grid'])
            if noise_type == 'bars':
                for x in range(0, 300, random.randint(10, 30)):
                    draw.rectangle([x, 0, x + random.randint(5, 15), 300], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            elif noise_type == 'circles':
                for _ in range(random.randint(5, 15)):
                    r = random.randint(10, 50)
                    x, y = random.randint(0, 300), random.randint(0, 300)
                    draw.ellipse([x-r, y-r, x+r, y+r], outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
            elif noise_type == 'waves':
                points = [(x, int(150 + 50 * np.sin(x / 20.0 + random.random()))) for x in range(0, 300, 5)]
                draw.line(points, fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=random.randint(2, 6))

            if random.random() > 0.4:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

            img.save(filepath)

if __name__ == '__main__':
    print("Starting Optimized Dataset Generation...")
    setup_directories()
    fetch_and_generate_candlesticks()
    generate_unknowns()
    print("Dataset generation complete! Ready for optimized training.")