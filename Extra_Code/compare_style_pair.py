"""
Given two handwriting images (or two folders of images, one per writer),
compute how close/far their handwriting *style* is according to the
trained style encoder.

Run from inside your Diffusion_HW_Hindi repo directory (so `train.py` is
importable).

-------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------
1. Requirements (install once):
       pip install torch torchvision numpy matplotlib pillow

2. Make sure `updated_diff_brush.py` (with Diffusion, TIMESTEPS, alphas,
   alpha_bars, betas, device, StyleEncoder) is importable, i.e. either:
       - it sits in the same folder as this script, or
       - its folder is on your PYTHONPATH.
   Run the script from inside your repo directory so this import resolves.

3. Point --ckpt at your trained checkpoint (.pt) that contains a
   "style_encoder" state dict.

4. Run one of:

    # single image vs single image
    python compare_style_pair.py \
        --ckpt /home/kishan/diffusion/diff_checkpoints_5/ckpt_epoch_599.pt \
        --img1 sample_images/writer_3/0.png \
        --img2 sample_images/writer_7/0.png

    # folder vs folder (averages embeddings across each writer's samples --
    # more robust than a single image, since one sample can be noisy)
    python compare_style_pair.py \
        --ckpt /home/kishan/diffusion/diff_checkpoints_5/ckpt_epoch_599.pt \
        --img1 sample_images/writer_3 \
        --img2 sample_images/writer_7

   Optional flags:
       --max_images N   cap number of images used per writer (if a folder
                         is passed with many images)
       --out PATH        where to save the comparison plot
                         (default: style_pair_comparison.png)

5. Output:
       - Printed table of cosine similarity / euclidean distance per
         embedding branch (global / ver / hor), plus a plain-language
         verdict based on the global embedding.
       - A saved PNG (--out) showing sample images side by side and a
         bar chart of the similarity scores.
-------------------------------------------------------------------------
"""

import os
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from updated_diff_brush import (   
    Diffusion,
    TIMESTEPS,
    alphas,
    alpha_bars,
    betas,
    device,
    StyleEncoder,
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def load_style_encoder(ckpt_path):
    model = StyleEncoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["style_encoder"])
    model.eval()
    print(f"[LOADED] style_encoder from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def list_images(path):
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(IMG_EXTS))
    return [path]


@torch.no_grad()
def embed_path(model, path, transform, max_images=None):
    """Returns averaged (global, ver, hor) embeddings over all images found
    at `path` (a single image or a folder), plus the per-image count."""
    files = list_images(path)
    if not files:
        raise ValueError(f"No images found at {path}")
    if max_images is not None:
        files = files[:max_images]

    global_embs, ver_embs, hor_embs = [], [], []
    for f in files:
        img = Image.open(f).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        _, _, ver_emb, hor_emb, global_emb = model(x)
        global_embs.append(global_emb.squeeze(0).cpu().numpy())
        ver_embs.append(ver_emb.squeeze(0).cpu().numpy())
        hor_embs.append(hor_emb.squeeze(0).cpu().numpy())

    return {
        "global": np.stack(global_embs).mean(axis=0),
        "ver": np.stack(ver_embs).mean(axis=0),
        "hor": np.stack(hor_embs).mean(axis=0),
        "n_images": len(files),
        "files": files,
    }


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def euclidean_dist(a, b):
    return float(np.linalg.norm(a - b))


def verdict(sim):
    if sim > 0.9:
        return "very similar handwriting style"
    elif sim > 0.75:
        return "fairly similar style"
    elif sim > 0.5:
        return "somewhat different style"
    else:
        return "clearly different style"


def plot_comparison(e1, e2, path1, path2, sims, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                              gridspec_kw={"width_ratios": [1, 1.4]})

    # left: representative images side by side
    ax_img = axes[0]
    try:
        im1 = Image.open(e1["files"][0]).convert("RGB")
        im2 = Image.open(e2["files"][0]).convert("RGB")
        h = 120
        im1 = im1.resize((int(im1.width * h / im1.height), h))
        im2 = im2.resize((int(im2.width * h / im2.height), h))
        combo_w = max(im1.width, im2.width)
        combo = Image.new("RGB", (combo_w, h * 2 + 10), (255, 255, 255))
        combo.paste(im1, (0, 0))
        combo.paste(im2, (0, h + 10))
        ax_img.imshow(combo)
    except Exception as ex:
        ax_img.text(0.5, 0.5, f"(preview unavailable: {ex})",
                    ha="center", va="center")
    ax_img.axis("off")
    ax_img.set_title("Writer 1 (top) vs Writer 2 (bottom)", fontsize=9)

    # right: bar chart of similarity by embedding branch
    ax_bar = axes[1]
    labels = list(sims.keys())
    values = [sims[k] for k in labels]
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in values]
    ax_bar.bar(labels, values, color=colors)
    ax_bar.axhline(0, color="black", linewidth=0.6)
    ax_bar.set_ylim(-1, 1)
    ax_bar.set_ylabel("cosine similarity")
    ax_bar.set_title(f"global: {sims['global']:.3f} -- {verdict(sims['global'])}",
                     fontsize=9)
    for i, v in enumerate(values):
        ax_bar.text(i, v + (0.03 if v >= 0 else -0.06), f"{v:.3f}",
                    ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--img1", required=True,
                    help="path to a single image, or a folder of images for writer 1")
    p.add_argument("--img2", required=True,
                    help="path to a single image, or a folder of images for writer 2")
    p.add_argument("--max_images", type=int, default=None,
                    help="cap images used per writer if a folder is passed")
    p.add_argument("--out", default="style_pair_comparison.png")
    args = p.parse_args()

    model = load_style_encoder(args.ckpt)
    transform = get_transform()

    e1 = embed_path(model, args.img1, transform, args.max_images)
    e2 = embed_path(model, args.img2, transform, args.max_images)

    sims = {
        "global": cosine_sim(e1["global"], e2["global"]),
        "ver": cosine_sim(e1["ver"], e2["ver"]),
        "hor": cosine_sim(e1["hor"], e2["hor"]),
    }
    dists = {
        "global": euclidean_dist(e1["global"], e2["global"]),
        "ver": euclidean_dist(e1["ver"], e2["ver"]),
        "hor": euclidean_dist(e1["hor"], e2["hor"]),
    }

    print(f"\nWriter 1: {args.img1}  ({e1['n_images']} image(s))")
    print(f"Writer 2: {args.img2}  ({e2['n_images']} image(s))")
    print(f"\n{'branch':<8}{'cosine sim':>12}{'euclid dist':>14}")
    for k in ("global", "ver", "hor"):
        print(f"{k:<8}{sims[k]:>12.4f}{dists[k]:>14.4f}")
    print(f"\nVerdict (based on global_emb): {verdict(sims['global'])}")
    print("(global_emb is the writer-identity embedding -- use it as the "
          "primary signal; ver/hor are content-masked proxy heads and are "
          "more sensitive to the specific glyphs in the images)")

    plot_comparison(e1, e2, args.img1, args.img2, sims, args.out)


if __name__ == "__main__":
    main()
