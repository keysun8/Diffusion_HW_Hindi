# Diffusion_HW_Hindi — Devanagari Handwriting Generation

A DiffBrush-style latent diffusion model that generates realistic **Devanagari (Hindi) handwriting** in a target writer's style from arbitrary input text.

## Table of Content
  * [Demo](#-demo)
  * [Upcoming](#upcoming)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Results](#results)
  * [Training Strategy](#training-strategy)
  * [Installation](#installation)
  * [Inference](#inference)
  * [Pretrained Weights](#pretrained-weights)
  * [Training](#training)
  * [Dataset & Latent Preparation](#dataset--latent-preparation)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Credits](#credits)

## 🎬 Demo

Two ways to try the model — pick whichever fits what you need:

<a href="https://huggingface.co/spaces/keysun89/HW_Hindi_Demo" target="_blank">
  <img align="right" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" width="150" alt="Try it on Hugging Face">
</a>

**1. Hugging Face Space** 👉 **[Launch Demo](https://huggingface.co/spaces/keysun89/HW_Hindi_Demo)**
Runs on free CPU hardware, so a single generation (1000 denoising steps) can take a couple of minutes — good for a quick, no-setup preview.

**2. Google Colab (recommended — GPU, much faster)** 👉 **[Open In Colab](https://colab.research.google.com/github/keysun8/Diffusion_HW_Hindi/blob/main/Diffusion_HW_Inference_Hindi.ipynb)**
Runs the same inference pipeline on a free Colab GPU. Clones this repo, downloads the checkpoint and VAE automatically, and lets you type any English (auto-transliterated) or Devanagari text.

<p align="center">
  <img src="https://github.com/keysun8/Diffusion_HW_Hindi/blob/main/HF_DEMO.png" alt="Devanagari Handwriting Generation Demo" width="700">
</p>
<p align="center">
  <em>Same text, rendered in different writers' handwriting styles by the diffusion model.</em>
</p>

## Upcoming

- **English handwriting generation** — Currently, the model is trained and released only for Hindi (Devanagari script). Over the next 1–2 weeks, I'll be releasing an English-language checkpoint trained on the same pipeline, along with updated weights and a demo on Hugging Face.

## Overview

This project trains a Stable Diffusion–based latent diffusion model for style-conditioned Devanagari handwriting generation. Given a style reference image from any writer and an arbitrary input text, the model generates a new image of that text rendered in the same handwriting style — effectively letting the model "mimic" a person's handwriting on unseen content. A style encoder extracts writer-identity features from the reference image, a content encoder encodes the target text, and a UNet-based latent diffusion model conditioned on both denoises a VAE latent that is decoded into the final handwriting image.

## Motivation

The motivation behind this project is twofold. First, the generated handwriting images can serve as a synthetic dataset — usable for downstream OCR training, where labeled handwriting data (especially in Hindi/Devanagari) is scarce and expensive to collect. Second, and more importantly, my primary goal was learning: understanding how style-content decoupled diffusion models work at a deep level. Alongside the core diffusion pipeline, I also experimented with a hybrid approach combining Autoregressive and Diffusion methods to compare generation quality and training dynamics. The same style-mimicking strategy is not limited to handwriting — it generalizes naturally to audio: given a reference audio sample of a person's voice and new text, a similar model could generate speech in that person's voice with the target content. The core learning here — decoupling "style/identity" from "content" and recombining them through diffusion — is the real takeaway of this project.

## Technical Aspect

- **Framework:** The current implementation is built on a custom PyTorch training loop. I'm planning to migrate parts of the framework to improve modularity, training speed, and ease of experimentation — this includes revisiting the trainer abstraction and exploring more efficient sampling schedules going forward.
- **Style Encoder:** A MobileNetV2 backbone feeds two parallel branches (`Ver_Style` / `Hor_Style`), each a stack of residual + self-attention blocks, producing full spatial vertical/horizontal style feature maps for conditioning, plus separately masked-and-pooled embeddings used only for metric learning.
- **Content-decoupled style learning:** Column-wise and row-wise random masking (DiffBrush's core idea) is applied only inside the Proxy-NCA embedding heads — never on the maps that feed the diffusion conditioning — forcing the style embeddings to capture writer identity rather than the specific glyphs in the reference image.
- **Content Encoder:** Renders input text to an image using a Devanagari TTF font (with RAQM shaping for correct matra/conjunct positioning), then encodes it through a MobileNetV2 backbone + learned positional embedding + self-attention stack into a content query `Q`.
- **Blender:** Fuses the content query with the vertical and horizontal style maps via sequential cross-attention (content → vertical style → horizontal style), producing the final conditioning signal for the UNet.
- **Diffusion UNet:** A residual/attention UNet operating in VAE latent space, conditioned on the diffusion timestep (sinusoidal embedding) and the blended style-content signal (cross-attention at every resolution). Uses independent width/height up- and down-sampling to match the wide, short aspect ratio of handwritten text lines.
- **Losses:** Standard DDPM noise-prediction MSE for the diffusion objective, plus three auxiliary Proxy-NCA losses (vertical style, horizontal style, global) so writer identity is explicitly discriminative in the learned embedding space.
- **Latent space:** All diffusion happens in a pretrained VAE's latent space (not pixel space), keeping training and sampling compute-efficient.
- **Sampling:** Standard ancestral DDPM denoising over the full 1000-step schedule, followed by VAE decoding back to an RGB image.

## Results

### Quantitative Metrics

Evaluated on the training split of writers:

| Metric | Train |
|---|---|
| HWD ↓ | 0.9698 |
| FID ↓ | 17.00 |
| IS ↑ | 2.24 ± 0.03 |
| GS ↓ | 0.0000913* |

*(↓ lower is better, ↑ higher is better)*

### Sample Generations

<p align="center">
  <img src="https://github.com/keysun8/Diffusion_HW_Hindi/blob/main/Hindi_HW_Imgs/Train_Imgs/3_train/10.jpg" alt="Generated samples — train writers" width="700">
</p>
<p align="center"><em>Generated handwriting conditioned on writers seen during training.</em></p>

<p align="center">
  <img src="https://github.com/keysun8/Diffusion_HW_Hindi/blob/main/Hindi_HW_Imgs/Test-Imgs/3_test/11.jpg" alt="Generated samples — test writers" width="700">
</p>
<p align="center"><em>Generated handwriting conditioned on held-out (unseen) test writers.</em></p>

## Training Strategy
The model was trained on approximately 900 unique writers, with around 25,000 sentence-level handwriting samples, using a dual-optimizer setup (style encoder trained via Proxy-NCA loss; content encoder + UNet trained via a combination of Proxy-NCA and MSE), mixed-precision training with gradient clipping, and per-module learning rates (content encoder/blender trained faster than the UNet). The final model achieved an FID of 17.00 and an HWD of 0.9698 on the training split, indicating strong visual fidelity and style consistency across writers.

## Installation

The code is written in Python 3.10+. If you don't have Python installed, you can find it [here](https://www.python.org/downloads/). Make sure you have the latest version of `pip` before installing dependencies.

Clone the repository and install the required packages:
```bash
git clone https://github.com/keysun8/Diffusion_HW_Hindi.git
cd Diffusion_HW_Hindi
pip install -r requirements.txt
```

A CUDA-enabled GPU is strongly recommended for both training and inference (sampling requires 1000 sequential UNet forward passes), though the scripts fall back to CPU automatically if none is available.

> **Note:** Devanagari shaping (matras, conjuncts) requires Pillow's RAQM layout engine. Modern Pillow wheels on Linux ship with `libraqm` already built in, so no extra system packages are usually needed — if you see incorrectly ordered Devanagari glyphs in the output, that's the signal your Pillow build is missing RAQM support.

## Inference

Download a trained checkpoint (see [Pretrained Weights](#pretrained-weights)) and the VAE, then generate an image:
```bash
python inference.py \
    --ckpt checkpoints/ckpt_epoch_0579.pt \
    --style_folder sample_images/writer_3 \
    --font_path NotoSansDevanagari-Regular.ttf \
    --vae_path ./vae \
    --text "namaste"
```

If `--text` is omitted, the script prompts for it interactively. Plain English input is automatically transliterated to Devanagari (ITRANS scheme); native Devanagari text is used as-is.

| Argument | Description | Default |
|---|---|---|
| `--ckpt` | Path to a trained checkpoint (`.pt`) | *required* |
| `--style_folder` | Folder of reference style images to sample from | *required* |
| `--style_image` | Use one specific style image instead of a random pick from `--style_folder` | `None` |
| `--text` | Text to render (English/ITRANS or Devanagari) | prompts interactively |
| `--out_dir` | Directory to save the generated image | `./inference_out` |
| `--out_name` | Output filename | random `output_img_<N>.png` |
| `--font_path` | Path to the Devanagari `.ttf` font | see script default |
| `--vae_path` | Path to the pretrained VAE (local folder or HF repo id) | see script default |
| `--img_height` / `--img_width` | Style-image resize dimensions | `64` / `1024` |
| `--seed` | Random seed for reproducible sampling | `None` |

The generated image is saved as a PNG inside `--out_dir`.

## Pretrained Weights

**Checkpoints** are hosted on Hugging Face Hub:
👉 **[keysun89/HW_Hindi_Model](https://huggingface.co/keysun89/HW_Hindi_Model/tree/main)**

```bash
huggingface-cli download keysun89/HW_Hindi_Model ckpt_epoch_0579.pt --local-dir checkpoints
```

**VAE weights** are hosted on Google Drive (too large for a Git repo):
👉 **[Download VAE folder](https://drive.google.com/drive/folders/1j7OpT5KYCkMf1oXEE9efufgB7lXAk17F)**

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1j7OpT5KYCkMf1oXEE9efufgB7lXAk17F -O ./vae
```

Point `--ckpt` / `--resume_ckpt` and `--vae_path` at the downloaded locations.

## Training

Train the full model (style encoder, content encoder, blender, diffusion UNet) jointly on pre-encoded handwriting latents:
```bash
python train.py \
    --data_root data/latents/train \
    --style_base_dir data/style_images/train \
    --vae_path ./vae \
    --batch_size 8 \
    --num_epochs 800 \
    --checkpoint_every 5
```

### Resuming training from a checkpoint
```bash
python train.py \
    --data_root data/latents/train \
    --style_base_dir data/style_images/train \
    --vae_path ./vae \
    --resume_ckpt checkpoints/ckpt_epoch_0369.pt \
    --num_epochs 800
```
Model weights, Proxy-NCA heads, the style optimizer, and both AMP scalers are restored exactly. The diffusion optimizer is intentionally re-initialized on resume with a learning-rate split (content encoder/blender train faster than the UNet), so its Adam momentum/variance resets while the underlying weights stay untouched.

| Argument | Description | Default |
|---|---|---|
| `--data_root` | Root directory of the latent training dataset (one subfolder per writer) | see script default |
| `--latent_ext` | File extension of stored latents (`.pt` or `.npy`) | `.pt` |
| `--vae_path` | Path to the pretrained VAE | see script default |
| `--batch_size` | Training batch size | `8` |
| `--num_workers` | DataLoader worker count | `4` |
| `--num_epochs` | Total number of epochs to train for | `800` |
| `--checkpoint_every` | Save a checkpoint every N epochs | `5` |
| `--checkpoint_dir` | Directory to save checkpoints to | `diff_checkpoints_5` |
| `--resume_ckpt` | Path to a checkpoint `.pt` file to resume training from | `None` |
| `--style_base_dir` | Base directory of per-writer style-image folders (used for training previews) | see script default |
| `--font_path` | Path to the Devanagari `.ttf` font used for content rendering | see script default |

Loss curves (`loss_curves.png` — diffusion MSE, vertical/horizontal/global Proxy-NCA) and per-epoch/mid-epoch sample generations are saved automatically as training progresses.

## Dataset & Latent Preparation

Training expects `--data_root` to contain one subfolder per writer:
```
data/latents/train/
├── writer_001/
│   ├── labels.txt      # one line of ground-truth text per sample, in order
│   ├── 0.pt            # pre-encoded VAE latent for sample 0
│   ├── 1.pt
│   └── ...
├── writer_002/
│   └── ...
```
The number of latent files in each writer folder must match the number of lines in `labels.txt`; mismatched writers are skipped automatically with a warning. Latents should be pre-encoded with the same VAE used everywhere else in the pipeline (`--vae_path`) so encoding is consistent between training and inference.

`--style_base_dir` follows a similar per-writer layout, but with raw style **images** (not latents) — these are sampled during training to produce qualitative preview generations, and at inference time via `--style_folder`.

> **Note:** The released model was trained on **900 unique writers**. This repo only includes a small subset of writer folders (under the style-images / sample data path) so people can run inference and see the expected folder structure — it is **not** the full training set.

## Technologies Used

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Diffusers-FFD21E?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

- **Deep Learning** — CNN and attention-based architectures (MobileNetV2, UNet, transformer-style self/cross-attention)
- **Diffusion Models** — DDPM-based forward/reverse noise process for image generation
- **Neural Network Training** — mixed-precision (AMP) training, gradient clipping, dual optimizers with per-module learning rates, and metric learning (Proxy-NCA)
- **PyTorch** — core deep learning framework
- **🤗 Diffusers** — pretrained VAE (`AutoencoderKL`) for latent encoding/decoding
- **Torchvision** — MobileNetV2 backbones for the style and content encoders
- **Pillow (+ RAQM)** — Devanagari font rendering with correct matra/conjunct shaping
- **indic_transliteration** — English (ITRANS) → Devanagari text transliteration
- **tqdm / Matplotlib** — training progress and loss-curve visualization
- **Hugging Face Hub & Spaces** — checkpoint hosting and the live CPU demo
- **Google Colab** — free-GPU inference notebook

## Team

**Kishan Madlani (Keysun)**
GitHub: [@keysun8](https://github.com/keysun8) · Hugging Face: [@keysun89](https://huggingface.co/keysun89)

## License

This project is licensed under the [MIT License](LICENSE).

## Credits

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) — foundational DDPM formulation used for the noise schedule and training objective
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752) — latent diffusion approach this project builds on
- [Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation (Dai et al., ICCV 2025)](https://arxiv.org/abs/2508.03256) — the content-decoupled style learning strategy (column/row masking, vertical + horizontal style branches) this project's style encoder is directly based on
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al., 2018)](https://arxiv.org/abs/1801.04381) — backbone architecture used in the style and content encoders
- [indic_transliteration](https://github.com/indic-transliteration/indic_transliteration_py) — ITRANS ↔ Devanagari transliteration library
- **CVIT Lab, IIIT Hyderabad** — model trained using GPU compute provided by the Centre for Visual Information Technology (CVIT) server
