"""
Visualize the style-encoder embedding space for DiffBrush-Hindi.

Run from inside your Diffusion_HW_Hindi repo directory (so `train.py` is
importable). Loads a checkpoint's style_encoder, embeds every image in a
style folder (or per-writer subfolders), and produces:

  1. A clean static PNG (points mode for a handful of writers/images,
     centroid mode for many writers -- auto-picked, override with --mode).
     Plots use scatter points only; no boundaries, polygons, or connecting lines.
  2. A SINGLE interactive HTML file (plotly) with writers split into a
     grid of subplot panels of --writers_per_plot (default 100) each --
     panel 1 covers writers 1-100, panel 2 covers 101-200, and so on.
     Each panel contains scatter points only.
  3. A writer-level (not image-level) cosine similarity heatmap.

-------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------
1. Requirements (install once):
       pip install torch torchvision numpy matplotlib pillow scikit-learn tqdm
       pip install plotly --break-system-packages   # needed for the HTML output

2. Make sure `updated_diff_brush.py` (with Diffusion, TIMESTEPS, alphas,
   alpha_bars, betas, device, StyleEncoder) is importable, i.e. either:
       - it sits in the same folder as this script, or
       - its folder is on your PYTHONPATH.
   Run the script from inside your repo directory so this import resolves.

3. Point --ckpt at your trained checkpoint (.pt) containing a
   "style_encoder" state dict, and --style_root at either:
       - a flat folder of images (treated as a single style/writer), or
       - a folder containing one subfolder per writer.

4. Run one of:

    # single style folder, all points are one writer
    python visualize_style_embeddings.py \
        --ckpt checkpoints/ckpt_epoch_0579.pt \
        --style_root sample_images/writer_3

    # many writer subfolders -- centroid view + chunked interactive html
    python visualize_style_embeddings.py \
        --ckpt /home/kishan/diffusion/diff_checkpoints_5/ckpt_epoch_599.pt \
        --style_root /home/kishan/diffusion/Syn_Data/syn_data_grouped \
        --max_writers 40 --max_per_writer 40 --writers_per_plot 40

   Useful optional flags:
       --embed_type {global,ver,hor,ver_hor}  (default: global — the
             writer-identity embedding; use this for "how close/far are
             writers")
       --method {tsne,pca}         (default: tsne)
       --mode {auto,points,centroids}  (default: auto)
       --max_writers N             cap number of writer subfolders used
       --max_per_writer N          cap images per writer
       --writers_per_plot N        writers per interactive-html panel
                                    (0 = one big html with every writer)
       --out_prefix PREFIX         prefix for all output filenames

5. Output (all written to the current directory, named from --out_prefix):
       - {prefix}_{method}.png (or _centroids.png)  -> static scatter plot
       - {prefix}_{method}.html (or _writers_combined.html) -> interactive plot
       - {prefix}_{method}_reduced.npz  -> cached 2-D coords/labels/names,
         so you can re-plot without rerunning t-SNE/PCA
       - {prefix}_similarity.png  -> writer-level (or image-level) cosine
         similarity heatmap
-------------------------------------------------------------------------
"""

import os
import argparse
import re

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon as MplPolygon
from PIL import Image
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

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


def natural_sort_key(name):
    """Sort numeric writer folder names as 1, 2, 3, ..., 10, 11, ..."""
    parts = re.split(r"(\\d+)", str(name))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# Model / embedding


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


@torch.no_grad()
def embed_folder(model, folder, transform, embed_type, max_per_writer=None):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))
    if max_per_writer is not None:
        files = files[:max_per_writer]
    embs, names = [], []
    for f in tqdm(files, desc=f"  Embedding {os.path.basename(folder)}",
                  leave=False, unit="img", dynamic_ncols=True):
        img = Image.open(os.path.join(folder, f)).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        _, _, ver_emb, hor_emb, global_emb = model(x)
        e = {"global": global_emb, "ver": ver_emb, "hor": hor_emb,
             "ver_hor": torch.cat([ver_emb, hor_emb], dim=1)}[embed_type]
        embs.append(e.squeeze(0).cpu().numpy())
        names.append(f)
    return (np.stack(embs) if embs else np.zeros((0, 512))), names


def collect_embeddings(model, style_root, transform, embed_type,
                        max_writers=None, max_per_writer=None):
    subdirs = [
        d for d in os.listdir(style_root)
        if os.path.isdir(os.path.join(style_root, d))
    ]
    subdirs = sorted(subdirs, key=natural_sort_key)

    if subdirs:
        if max_writers is not None:
            subdirs = subdirs[:max_writers]

        print(
            f"[INFO] Selected writers: {subdirs[0]} -> {subdirs[-1]} "
            f"({len(subdirs)} folders)"
        )
        all_embs, labels, names = [], [], []
        for d in tqdm(subdirs, desc="Writers", unit="writer",
                      dynamic_ncols=True):
            embs, fnames = embed_folder(model, os.path.join(style_root, d),
                                         transform, embed_type, max_per_writer)
            if len(embs) == 0:
                continue
            all_embs.append(embs)
            labels += [d] * len(embs)
            names += [f"{d}/{n}" for n in fnames]
        embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 512))
        return embs, labels, names, True  # grouped=True
    else:
        embs, fnames = embed_folder(model, style_root, transform, embed_type,
                                     max_per_writer)
        return embs, ["style"] * len(embs), fnames, False  # grouped=False


# Dimensionality reduction

def reduce_2d(embs, method):
    n = embs.shape[0]
    if method == "tsne":
        perplexity = max(2, min(30, n // 3))
        reducer = TSNE(n_components=2, perplexity=perplexity, init="pca",
                        random_state=42)
    else:
        reducer = PCA(n_components=2)
    return reducer.fit_transform(embs)


# Static plots


def plot_points(coords, labels, out_path, title):
    """Scatter-only static plot. No boundaries, polygons, lines, or ellipses."""
    fig, ax = plt.subplots(figsize=(15, 12), dpi=220)
    uniq = sorted(set(labels), key=natural_sort_key)
    cmap = plt.get_cmap("turbo", max(len(uniq), 2))

    for i, lab in enumerate(uniq):
        idx = np.flatnonzero(np.asarray(labels) == lab)
        pts = coords[idx]
        color = cmap(i / max(len(uniq) - 1, 1))

        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=18, alpha=0.70, color=color, linewidth=0,
            rasterized=True, zorder=2
        )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")

def plot_centroids(coords, labels, out_path, title):
    """Scatter-only writer-centroid plot. No ellipses or boundaries."""
    uniq = sorted(set(labels), key=natural_sort_key)
    fig, ax = plt.subplots(figsize=(18, 14), dpi=220)
    cmap = plt.get_cmap("turbo", max(len(uniq), 2))

    for i, lab in enumerate(uniq):
        idx = np.flatnonzero(np.asarray(labels) == lab)
        pts = coords[idx]
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        color = cmap(i / max(len(uniq) - 1, 1))

        ax.scatter(
            cx, cy, s=38, color=color, edgecolor="white",
            linewidth=0.45, alpha=0.90, zorder=3,
            rasterized=True
        )

        if len(uniq) <= 100:
            ax.annotate(
                lab, (cx, cy), fontsize=7, xytext=(3, 3),
                textcoords="offset points", alpha=0.85, zorder=4
            )

    ax.set_title(
        f"{title}\n{len(uniq)} writers — centroid view",
        fontsize=19, fontweight="bold", pad=16
    )
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")

def plot_similarity_matrix(embs, labels, out_path, title):
    """Group-level (writer-level) cosine similarity, using the mean
    embedding per group -- readable even with hundreds of writers,
    unlike an image-level matrix."""
    uniq = sorted(set(labels), key=natural_sort_key)
    centroids = np.stack([
        embs[[i for i, l in enumerate(labels) if l == lab]].mean(axis=0)
        for lab in uniq
    ])
    c = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sim = c @ c.T
    n = len(uniq)
    # Keep the canvas bounded for large writer sets (e.g. 900 x 900).
    # The matrix itself is still full-resolution; only the figure dimensions
    # are capped to avoid creating an enormous image.
    fig_size = min(18, max(8, n * 0.018))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=220)
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="coolwarm")
    show_ticks = n <= 60
    if show_ticks:
        ax.set_xticks(range(n)); ax.set_xticklabels(uniq, rotation=90, fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(uniq, fontsize=6)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# Interactive plot(s)


def chunk_writers(labels, chunk_size):
    """Split naturally sorted writer labels into consecutive chunks."""
    uniq = sorted(set(labels))
    return [uniq[i:i + chunk_size] for i in range(0, len(uniq), chunk_size)]


def plot_interactive(coords, labels, names, out_path, title):
    try:
        import plotly.express as px
    except ImportError:
        print("[SKIP] interactive html needs plotly: "
              "pip install plotly --break-system-packages")
        return
    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1],
        color=labels, hover_name=names,
        title=title, opacity=0.75,
        width=1100, height=850,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.write_html(out_path)
    print(f"[SAVED] {out_path}  (open in a browser -- hover/zoom/click legend to isolate a writer)")


# A largish qualitative palette so writers within a subplot get visually
# distinct colors even with ~100 groups per panel (colors repeat only
# after this many distinct writers within a single subplot).
_QUALITATIVE_PALETTE = None


def _get_palette():
    global _QUALITATIVE_PALETTE
    if _QUALITATIVE_PALETTE is None:
        import plotly.colors as pc
        _QUALITATIVE_PALETTE = (pc.qualitative.Alphabet
                                 + pc.qualitative.Dark24
                                 + pc.qualitative.Light24)
    return _QUALITATIVE_PALETTE


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plot_interactive_chunked(coords, labels, names, out_prefix, title,
                              writers_per_plot, grouped):
    """Interactive scatter-only HTML. No boundaries, lines, or filled regions."""
    if not grouped or writers_per_plot is None or writers_per_plot <= 0:
        plot_interactive(coords, labels, names, f"{out_prefix}.html", title)
        return

    chunks = chunk_writers(labels, writers_per_plot)
    n_chunks = len(chunks)

    if n_chunks <= 1:
        plot_interactive(coords, labels, names, f"{out_prefix}.html", title)
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[SKIP] interactive html needs plotly: "
              "pip install plotly --break-system-packages")
        return

    palette = _get_palette()

    cols = min(3, n_chunks)
    rows = (n_chunks + cols - 1) // cols

    subplot_titles = []
    for plot_idx in range(1, n_chunks + 1):
        start_n = (plot_idx - 1) * writers_per_plot + 1
        end_n = start_n + len(chunks[plot_idx - 1]) - 1
        subplot_titles.append(f"Writers {start_n}-{end_n}")

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.06
    )

    for plot_idx, chunk in enumerate(chunks, start=1):
        r = (plot_idx - 1) // cols + 1
        c = (plot_idx - 1) % cols + 1

        for w_i, writer in enumerate(chunk):
            idx = [i for i, l in enumerate(labels) if l == writer]
            if not idx:
                continue

            color = palette[w_i % len(palette)]
            pts = coords[idx]

            # ONLY SCATTER POINTS — no hulls, polygons, lines, or fills.
            fig.add_trace(
                go.Scattergl(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="markers",
                    marker=dict(
                        size=6,
                        opacity=0.80,
                        color=color,
                        line=dict(width=0)
                    ),
                    text=[names[i] for i in idx],
                    hovertemplate=(
                        "%{text}<br>writer: " + writer + "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

    fig.update_layout(
        title=title,
        width=420 * cols,
        height=380 * rows,
    )

    out_path = f"{out_prefix}_writers_combined.html"
    fig.write_html(out_path)
    print(
        f"[SAVED] {out_path} "
        f"({n_chunks} panels of up to {writers_per_plot} writers each)"
    )


# Main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--style_root", required=True,
                    help="flat folder of images, OR a folder containing "
                         "one subfolder per writer")
    p.add_argument("--embed_type", choices=["global", "ver", "hor", "ver_hor"],
                    default="global",
                    help="global_emb is the writer-identity embedding to use "
                         "for 'how close/far are writers'; ver/hor are "
                         "content-masked proxy heads")
    p.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    p.add_argument("--mode", choices=["auto", "points", "centroids"], default="auto",
                    help="auto: points if <=20 groups, else centroids")
    p.add_argument("--max_writers", type=int, default=None,
                    help="select the first N writer subfolders after numeric/natural sorting")
    p.add_argument("--max_per_writer", type=int, default=None,
                    help="cap images per writer (also applies to a flat folder)")
    p.add_argument("--writers_per_plot", type=int, default=100,
                    help="split the interactive html into groups of this many "
                         "writers each (plot 1 = writers 1..N, plot 2 = next N, "
                         "...). Set to 0 to disable chunking and get one big "
                         "html with every writer.")
    p.add_argument("--out_prefix", default="style_embed")
    args = p.parse_args()
    model = load_style_encoder(args.ckpt)
    transform = get_transform()

    embs, labels, names, grouped = collect_embeddings(
        model, args.style_root, transform, args.embed_type,
        args.max_writers, args.max_per_writer)

    print(f"Total images embedded: {embs.shape[0]}, dim={embs.shape[1] if embs.size else 0}, "
          f"groups: {len(set(labels))}")
    if embs.shape[0] < 3:
        print("[ERROR] need at least 3 embedded images, check --style_root")
        return

    print(f"[INFO] Running {args.method.upper()} dimensionality reduction on "
          f"{embs.shape[0]:,} embeddings...")
    coords = reduce_2d(embs, args.method)
    print("[DONE] Dimensionality reduction complete.")

    # Save the 2-D representation so future plots do not require rerunning
    # the expensive t-SNE/PCA step.
    reduced_path = f"{args.out_prefix}_{args.method}_reduced.npz"
    np.savez_compressed(
        reduced_path,
        coords=coords.astype(np.float32),
        labels=np.asarray(labels, dtype=str),
        names=np.asarray(names, dtype=str),
        method=np.asarray(args.method),
        embed_type=np.asarray(args.embed_type),
    )
    print(f"[SAVED] {reduced_path}")

    n_groups = len(set(labels))
    mode = args.mode
    if mode == "auto":
        mode = "points"

    title = f"Style embedding space ({args.embed_type}, {args.method})"
    if mode == "points" or not grouped:
        plot_points(coords, labels, f"{args.out_prefix}_{args.method}.png", title)
    else:
        plot_centroids(coords, labels, f"{args.out_prefix}_{args.method}_centroids.png",
                       title + " -- per-writer centroids")

    plot_interactive_chunked(
        coords, labels, names,
        f"{args.out_prefix}_{args.method}",
        title,
        args.writers_per_plot,
        grouped,
    )

    if grouped:
        plot_similarity_matrix(embs, labels, f"{args.out_prefix}_similarity.png",
                               "Writer-level cosine similarity")
    else:
        # flat folder: image-level matrix is fine, folder is usually small
        plot_similarity_matrix(embs, list(range(len(embs))),
                               f"{args.out_prefix}_similarity.png",
                               "Image-level cosine similarity")


if __name__ == "__main__":
    main()
