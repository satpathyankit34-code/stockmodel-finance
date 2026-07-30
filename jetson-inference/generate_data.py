import os
import shutil
import random
from PIL import Image, ImageDraw
import pandas as pd
import yfinance as yf
import numpy as np
import mplfinance as mpf

base_dir = "python/training/classification/data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
classes = ['buy', 'hold', 'sell', 'unknown']

print("Clearing previous dataset files...")
for split_dir in [train_dir, val_dir]:
    if os.path.exists(split_dir):
        for c in classes:
            class_path = os.path.join(split_dir, c)
            if os.path.exists(class_path):
                shutil.rmtree(class_path)
            os.makedirs(class_path, exist_ok=True)
    else:
        for c in classes:
            os.makedirs(os.path.join(split_dir, c), exist_ok=True)

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", 
    "AMD", "INTC", "JPM", "V", "JNJ", "WMT", "DIS", "PYPL", "BA", 
    "SPY", "QQQ", "BTC-USD", "ETH-USD"
]

print("Generating historical candlestick chart dataset...")
total_generated = {c: 0 for c in classes}

# 1. Generate Financial Charts (Buy, Hold, Sell)
for ticker in tickers:
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty or len(df) < 60:
            continue
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            continue

        window_size = 40
        for i in range(window_size, len(df) - 10, 5):
            window = df.iloc[i-window_size:i].copy()
            future_return = (df['Close'].iloc[i+10] - df['Close'].iloc[i]) / df['Close'].iloc[i]
            
            if future_return > 0.03:
                label = 'buy'
            elif future_return < -0.03:
                label = 'sell'
            else:
                label = 'hold'
                
            split = 'train' if random.random() < 0.8 else 'val'
            theme = random.choice(['light', 'dark'])
            
            if theme == 'dark':
                mc = mpf.make_marketcolors(up='#00ffcc', down='#ff0066', edge='inherit', wick='inherit', volume='in')
                style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, figcolor='#121212', facecolor='#121212')
            else:
                mc = mpf.make_marketcolors(up='#008800', down='#cc0000', edge='inherit', wick='inherit', volume='in')
                style = mpf.make_mpf_style(base_mpf_style='default', marketcolors=mc, figcolor='white', facecolor='white')
                
            filename = f"{ticker}_{i}_{theme}_{random.randint(100,999)}.png"
            dest_path = os.path.join(base_dir, split, label, filename)
            
            mpf.plot(
                window,
                type='candle',
                style=style,
                savefig=dict(fname=dest_path, dpi=100, bbox_inches='tight', pad_inches=0),
                axisoff=True,
                scale_padding={'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
            )
            total_generated[label] += 1
    except Exception as e:
        pass

# 2. Generate robust 'unknown' distractors (complex patterns, fake UI, random shapes)
print("Generating advanced 'unknown' background/distractor class...")
target_unknown_count = sum(total_generated[c] for c in ['buy', 'hold', 'sell']) // 3

for i in range(target_unknown_count):
    split = 'train' if random.random() < 0.8 else 'val'
    filename = f"distractor_{i}_{random.randint(100,999)}.png"
    dest_path = os.path.join(base_dir, split, 'unknown', filename)
    
    try:
        # Create blank image
        img = Image.new('RGB', (600, 400), color=(random.randint(20,230), random.randint(20,230), random.randint(20,230)))
        draw = ImageDraw.Draw(img)
        
        distractor_style = random.choice(['geometric', 'lines', 'boxes', 'noise'])
        
        if distractor_style == 'geometric':
            for _ in range(random.randint(5, 15)):
                shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                box = [random.randint(0, 500), random.randint(0, 300), random.randint(100, 600), random.randint(100, 400)]
                if shape_type == 'rectangle':
                    draw.rectangle(box, fill=color)
                elif shape_type == 'ellipse':
                    draw.ellipse(box, fill=color)
        elif distractor_style == 'lines':
            for _ in range(random.randint(20, 50)):
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                draw.line([(random.randint(0, 600), random.randint(0, 400)), (random.randint(0, 600), random.randint(0, 400))], fill=color, width=random.randint(1, 5))
        elif distractor_style == 'boxes':
            # Simulate random UI elements or text boxes
            for _ in range(random.randint(4, 10)):
                box = [random.randint(10, 400), random.randint(10, 300), random.randint(50, 550), random.randint(50, 380)]
                draw.rectangle(box, outline=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), width=random.randint(2,8))
        else:
            arr = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            
        img.save(dest_path)
        total_generated['unknown'] += 1
    except Exception as e:
        pass

print("\n--- DATASET RE-GENERATION COMPLETE ---")
for c in classes:
    print(f"Class '{c}': {total_generated[c]} images generated.")
