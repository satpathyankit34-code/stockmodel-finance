import os
import shutil
import random

# Source where your generated images are currently stored
source_dir = "data/images"  # Adjust this if your images are directly in another folder
classes = ['buy', 'hold', 'sell', 'unknown']

# Target structure expected by train.py
base_data = "data"
for split in ['train', 'val']:
    for c in classes:
        os.makedirs(os.path.join(base_data, split, c), exist_ok=True)

for c in classes:
    class_path = os.path.join(source_dir, c)
    if not os.path.exists(class_path):
        print(f"Warning: Class folder {class_path} not found!")
        continue
        
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    
    split_idx = int(0.8 * len(images))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(base_data, 'train', c, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(base_data, 'val', c, img))
        
print("Dataset successfully reorganized into train/ and val/ splits!")
