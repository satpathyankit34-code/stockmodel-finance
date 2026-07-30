import os
import shutil
import random

base_dir = "python/training/classification/data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

classes = ['buy', 'hold', 'sell', 'unknown']

# 1. Ensure all target folders exist
for c in classes:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(val_dir, c), exist_ok=True)

# 2. Gather all image files from the workspace
image_extensions = ('.png', '.jpg', '.jpeg')
all_images = []

for root, dirs, files in os.walk("python/training/classification"):
    # Skip train and val dirs during collection to avoid duplicate re-adding
    if 'train' in root or 'val' in root:
        continue
    for file in files:
        if file.lower().endswith(image_extensions):
            all_images.append(os.path.join(root, file))

# Fallback if no images found outside, check everywhere
if not all_images:
    for root, dirs, files in os.walk("."):
        if 'docs' in root or '.git' in root:
            continue
        for file in files:
            if file.lower().endswith(image_extensions):
                all_images.append(os.path.join(root, file))

print(f"Found {len(all_images)} total images to organize.")

# 3. Assign classes based on filename keywords (or default to 'unknown')
categorized = {c: [] for c in classes}
for img_path in all_images:
    filename = os.path.basename(img_path).lower()
    assigned = False
    for c in classes:
        if c in filename:
            categorized[c].append(img_path)
            assigned = True
            break
    if not assigned:
        categorized['unknown'].append(img_path)

# 4. Split into train and val and copy
for c in classes:
    imgs = categorized[c]
    random.shuffle(imgs)
    split_idx = int(0.8 * len(imgs))
    
    train_subset = imgs[:split_idx]
    val_subset = imgs[split_idx:]
    
    for img in train_subset:
        dest = os.path.join(train_dir, c, os.path.basename(img))
        shutil.copy(img, dest)
        
    for img in val_subset:
        dest = os.path.join(val_dir, c, os.path.basename(img))
        shutil.copy(img, dest)
        
    print(f"Class '{c}': {len(train_subset)} train, {len(val_subset)} val")

print("Dataset organization complete!")
