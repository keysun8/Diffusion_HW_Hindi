"""
compute_is.py
Computes Inception Score for generated images (no labels/real data needed).

-------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------
1. Requirements (install once):
       pip install torch torchmetrics torchvision pillow tqdm

2. Update GEN_DIRS near the top of this file with the folder(s) of
   generated images you want to score. Each entry is searched recursively
   for .png files (see `glob.glob(f"{gen_dir}/**/*.png", recursive=True)`),
   so subfolders (e.g. one per writer/style) are fine.

3. Run it:
       python compute_is.py

   Output:
       - Printed image count per folder in GEN_DIRS
       - A tqdm progress bar while batches are scored
       - Printed Inception Score (mean ± std) per folder
-------------------------------------------------------------------------
"""

import torch
from torchmetrics.image.inception import InceptionScore
from PIL import Image
from torchvision import transforms
import glob
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEN_DIRS = [
    "/home/kishan/diffusion/Syn_Data/syn_data_grouped",
]

transform = transforms.Compose([
    transforms.Resize((299, 299)),   # InceptionV3 input size
    transforms.ToTensor(),
])

def load_images_as_uint8_batch(paths, batch_size=32):
    batch = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img_t = transform(img)
        img_uint8 = (img_t * 255).to(torch.uint8)
        batch.append(img_uint8)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
    if batch:
        yield torch.stack(batch)

def compute_is_for_dir(gen_dir):
    paths = glob.glob(f"{gen_dir}/**/*.png", recursive=True)
    print(f"{gen_dir}: {len(paths)} images")

    inception = InceptionScore().to(DEVICE)

    for batch in tqdm(load_images_as_uint8_batch(paths), desc=f"IS [{gen_dir}]"):
        inception.update(batch.to(DEVICE))

    mean, std = inception.compute()
    print(f"IS [{gen_dir}]: mean={mean.item():.4f}, std={std.item():.4f}")
    return mean.item(), std.item()

if __name__ == "__main__":
    for gen_dir in GEN_DIRS:
        compute_is_for_dir(gen_dir)
