import os
import shutil
import random

base_dir = "python/training/classification/data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

classes = ['buy', 'hold', 'sell', 'unknown']

# Ensure target directories exist
for c in classes:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(val_dir, c), exist_ok=True)

# Gather all images currently dumped in train/unknown or elsewhere
all_images = []
for c in classes:
    src_c_dir = os.path.join(train_dir, c)
    if os.path.exists(src_c_dir):
        for img in os.listdir(src_c_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append((img, c))
                
# If they are all lumped in unknown, redistribute them or split them properly
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

# Clear out old placements to re-distribute cleanly
for c in classes:
    for split_path in [train_dir, val_dir]:
        target_sub = os.path.join(split_path, c)
        for f in os.listdir(target_sub):
            os.remove(os.path.join(target_sub, f))

for img, c in train_imgs:
    src = os.path.join(train_dir, c, img) # Might need adjustment depending on origin
    # Fallback search if path shifted
    shutil.move(os.path.join(train_dir, 'unknown', img), os.path.join(train_dir, c, img))

print("Reorganization complete!")
