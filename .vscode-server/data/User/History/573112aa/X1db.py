import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from PIL import Image, ImageDraw, ImageFilter

# --- CONFIGURATION ---
TRAIN_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'SPY', 'QQQ', 'AMD', 'NFLX', 'DIS', 'INTC']
VAL_TICKERS = ['BA', 'PYPL', 'ADBE', 'CRM', 'PEP', 'COST']

OUTPUT_DIR = os.path.expanduser('~/financial_charts_data')
CLASSES = ['buy', 'hold', 'sell', 'unknown']

# Reduced target count for fast generation and testing
TARGET_IMAGES_PER_CLASS = 50
WINDOW_SIZE = 30

def setup_directories():
    print(f"Creating output directory at '{OUTPUT_DIR}'...")
    for split in ['train', 'val']:
        for cls in CLASSES:
            dir_path = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(dir_path, exist_ok=True)
    print("Directories verified and ready.")

def fetch_and_generate_candlesticks():
    print("Fetching market data and rendering charts...")
    
    counts = {
        'train': {'buy': 0, 'hold': 0, 'sell': 0},
        'val': {'buy': 0, 'hold': 0, 'sell': 0}
    }

    all_tickers = [(t, 'train') for t in TRAIN_TICKERS] + [(t, 'val') for t in VAL_TICKERS]

    for ticker, split in all_tickers:
        if all(counts[split][c] >= TARGET_IMAGES_PER_CLASS for c in ['buy', 'hold', 'sell']):
            break

        print(f"Downloading history for: {ticker} ({split})")
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period='max', interval='1d')
            
            if df.empty or len(df) < WINDOW_SIZE + 5:
                continue

            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                continue

            indices = list(range(len(df) - WINDOW_SIZE - 5))
            random.shuffle(indices)

            for i in indices:
                if all(counts[split][c] >= TARGET_IMAGES_PER_CLASS for c in ['buy', 'hold', 'sell']):
                    break

                window = df.iloc[i:i+WINDOW_SIZE].copy()
                
                future_window = df.iloc[i+WINDOW_SIZE : i+WINDOW_SIZE+5]
                future_max = future_window['High'].max()
                future_min = future_window['Low'].min()
                current_price = window.iloc[-1]['Close']

                max_gain = (future_max - current_price) / current_price
                max_loss = (future_min - current_price) / current_price

                if max_gain > 0.015 and max_gain > abs(max_loss):
                    label = 'buy'
                elif max_loss < -0.015 and abs(max_loss) > max_gain:
                    label = 'sell'
                else:
                    label = 'hold'

                if counts[split][label] >= TARGET_IMAGES_PER_CLASS:
                    continue

                for col in required_cols:
                    min_val = window[col].min()
                    max_val = window[col].max()
                    if max_val - min_val > 0:
                        window[col] = (window[col] - min_val) / (max_val - min_val) * 100.0
                    else:
                        window[col] = 50.0

                filename = f"{ticker}_{i}.png"
                filepath = os.path.join(OUTPUT_DIR, split, label, filename)

                style = random.choice(['binance', 'yahoo', 'charles'])
                mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
                s = mpf.make_mpf_style(marketcolors=mc, base_style=style)

                mpf.plot(
                    window, 
                    type='candle', 
                    style=s, 
                    axisoff=True, 
                    savefig=dict(fname=filepath, dpi=100, bbox_inches='tight', pad_inches=0.1)
                )

                counts[split][label] += 1
                print(f" -> [{split.upper()} / {label}]: {counts[split][label]} / {TARGET_IMAGES_PER_CLASS}")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

def generate_unknowns():
    print("Generating unknown/distractor images...")
    total_structured = TARGET_IMAGES_PER_CLASS * 3
    
    for split, ratio in [('train', 0.8), ('val', 0.2)]:
        unknown_target = int(total_structured * ratio)
        print(f"Generating {unknown_target} unknown images for {split}...")
        for i in range(unknown_target):
            filepath = os.path.join(OUTPUT_DIR, split, 'unknown', f"distractor_{i}.png")

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
    print("Starting Fast Lightweight Dataset Generation...")
    setup_directories()
    fetch_and_generate_candlesticks()
    generate_unknowns()
    print(f"Dataset generation complete! Files saved to: {OUTPUT_DIR}")