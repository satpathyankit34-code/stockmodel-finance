import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

# =========================================================
# CONFIGURATION
# =========================================================
OUTPUT_DIR = "dataset"
CLASSES = ["BUY", "SELL", "HOLD", "UNKNOWN"]
IMAGES_PER_CLASS = 1000
IMG_SIZE = (224, 224)

COLOR_UP = '#089981'    # TradingView Green
COLOR_DOWN = '#F23645'  # TradingView Red
BG_COLOR = 'white'

def setup_directories():
    for split in ['train', 'val']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# =========================================================
# UI FRAME EMBEDDER (Simulates Trading View / Webull Layout)
# =========================================================
def embed_in_ui_frame(viewport_img=None, is_unknown=False):
    """Wraps any content inside a realistic light-theme trading UI template."""
    canvas = Image.new('RGB', IMG_SIZE, color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 1. Top Navigation Bar (Header/Tabs)
    draw.rectangle([0, 0, 224, 20], fill=(245, 247, 250))
    draw.line([(0, 20), (224, 20)], fill=(225, 228, 232), width=1)
    draw.rectangle([8, 5, 45, 15], fill=(220, 225, 232)) # Fake Ticker Search
    draw.rectangle([52, 5, 80, 15], fill=(235, 238, 242)) # Fake Interval Button
    
    # 2. Left Toolbar (Drawing Tools)
    draw.rectangle([0, 20, 16, 224], fill=(248, 249, 250))
    draw.line([(16, 20), (16, 224)], fill=(225, 228, 232), width=1)
    
    # 3. Right Price Scale Axis
    draw.rectangle([196, 20, 224, 206], fill=(255, 255, 255))
    draw.line([(196, 20), (196, 206)], fill=(225, 228, 232), width=1)
    for y in range(32, 200, 24): # Price tick marks
        draw.line([(196, y), (200, y)], fill=(180, 185, 190))
        
    # 4. Bottom Time Scale Axis
    draw.rectangle([16, 206, 196, 224], fill=(255, 255, 255))
    draw.line([(16, 206), (196, 206)], fill=(225, 228, 232), width=1)
    for x in range(35, 180, 30): # Time tick marks
        draw.line([(x, 206), (x, 210)], fill=(180, 185, 190))

    # 5. Fill Middle Viewport (16, 20) to (196, 206)
    viewport_w = 196 - 16  # 180px
    viewport_h = 206 - 20  # 186px

    if not is_unknown and viewport_img is not None:
        # Paste chart canvas in the middle
        resized_chart = viewport_img.resize((viewport_w, viewport_h))
        canvas.paste(resized_chart, (16, 20))
    else:
        # Generate NON-CHART content inside the viewport (Watchlist, Text, Blank, Settings)
        mode = random.choice(['watchlist', 'text_lines', 'blank_panel'])
        if mode == 'watchlist':
            for y in range(30, 190, 16):
                draw.rectangle([25, y, 110, y + 8], fill=(220, 225, 230)) # Ticker name
                draw.rectangle([130, y, 180, y + 8], fill=(200, 230, 210) if random.random() > 0.5 else (245, 200, 200)) # Price change
        elif mode == 'text_lines':
            for y in range(30, 190, 12):
                x_end = random.randint(70, 180)
                draw.line([(25, y), (x_end, y)], fill=(210, 215, 220), width=2)
        elif mode == 'blank_panel':
            draw.rectangle([16, 20, 196, 206], fill=(252, 252, 253))

    return canvas

# =========================================================
# CHART GENERATORS (BUY, SELL, HOLD)
# =========================================================
def generate_synthetic_ohlc(num_candles, trend_type):
    prices = [100.0]
    for _ in range(num_candles - 1):
        if trend_type == "BUY":
            change = np.random.normal(0.4, 1.0)
        elif trend_type == "SELL":
            change = np.random.normal(-0.4, 1.0)
        else:  # HOLD
            change = np.random.normal(0.0, 0.8)
        prices.append(prices[-1] + change)
        
    opens, highs, lows, closes = [], [], [], []
    for p in prices:
        o = p + np.random.uniform(-0.5, 0.5)
        c = p + np.random.uniform(-0.5, 0.5)
        h = max(o, c) + np.random.uniform(0.1, 0.8)
        l = min(o, c) - np.random.uniform(0.1, 0.8)
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        
    return opens, highs, lows, closes

def render_chart_canvas(opens, highs, lows, closes):
    """Renders raw chart candles onto a PIL image."""
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=70)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    if random.random() > 0.4:
        ax.grid(True, color='#E5E7EB', linestyle='-', linewidth=0.5, alpha=0.7)
        
    for i in range(len(opens)):
        is_up = closes[i] >= opens[i]
        color = COLOR_UP if is_up else COLOR_DOWN
        
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1.2)
        height = max(abs(closes[i] - opens[i]), 0.1)
        bottom = min(opens[i], closes[i])
        rect = patches.Rectangle((i - 0.35, bottom), 0.7, height, 
                                 facecolor=color, edgecolor=color, fill=True)
        ax.add_patch(rect)
        
    ax.autoscale_view()
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    fig.canvas.draw()
    chart_pil = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return chart_pil

# =========================================================
# MAIN GENERATION LOOP
# =========================================================
def main():
    setup_directories()
    print("Generating full-UI context dataset...")
    
    for split in ['train', 'val']:
        count = IMAGES_PER_CLASS if split == 'train' else int(IMAGES_PER_CLASS * 0.2)
        print(f"Generating {split} set ({count} images per class)...")
        
        for cls in CLASSES:
            for i in range(count):
                file_path = os.path.join(OUTPUT_DIR, split, cls, f"{cls}_{i:04d}.png")
                
                if cls in ["BUY", "SELL", "HOLD"]:
                    num_candles = random.randint(20, 45)
                    o, h, l, c = generate_synthetic_ohlc(num_candles, cls)
                    chart_img = render_chart_canvas(o, h, l, c)
                    full_frame = embed_in_ui_frame(chart_img, is_unknown=False)
                else:  # UNKNOWN
                    full_frame = embed_in_ui_frame(viewport_img=None, is_unknown=True)
                
                full_frame.save(file_path)

    print("Full-UI dataset generation complete!")

if __name__ == "__main__":
    main()