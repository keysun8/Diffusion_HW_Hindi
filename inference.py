"""
inference.py
Standalone inference for the DiffBrush-style Devanagari handwriting model.

Usage:
    python inference.py \
        --ckpt /home/kishan/diffusion/diff_checkpoints_5/ckpt_epoch_579.pt \
        --style_folder /home/kishan/diffusion/output_dataset_hindi_with_json_line_64*1024/train/3 \
        --out_dir ./inference_out \
        --font_path /home/kishan/diffusion/font/NotoSansDevanagari-Regular.ttf \
        --vae_path /home/kishan/diffusion/vae
"""

import os
import argparse
import random

import torch
from PIL import Image
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from torch.amp import autocast
from torchvision.utils import save_image

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from train import (
    Diffusion,
    TIMESTEPS,
    alphas,
    alpha_bars,
    betas,
    device,
)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _load_partial(module, state_dict, name):
    """strict=False load -- tolerates old checkpoints missing/extra keys
    (e.g. pos_embed added later to ContentEncoder). Prints what didn't match."""
    result = module.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"[{name}] missing keys (kept freshly-initialized): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[{name}] unexpected keys in ckpt (ignored): {result.unexpected_keys}")


def load_model_for_inference(ckpt_path, font_path):
    model = Diffusion(font_path=font_path).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    _load_partial(model.style_encoder, ckpt["style_encoder"], "style_encoder")
    _load_partial(model.content_encoder, ckpt["content_encoder"], "content_encoder")
    _load_partial(model.blender, ckpt["blender"], "blender")
    _load_partial(model.unet, ckpt["unet"], "unet")
    _load_partial(model.final, ckpt["final"], "final")
    if "embedding_head" in ckpt:
        _load_partial(model.embedding_head, ckpt["embedding_head"], "embedding_head")

    model.eval()
    print(f"[LOADED] checkpoint from epoch {ckpt.get('epoch', '?')}")
    return model


def pick_style_image(style_folder, style_image=None):
    if style_image is not None:
        return style_image
    files = sorted(f for f in os.listdir(style_folder) if f.lower().endswith(IMG_EXTS))
    if not files:
        raise FileNotFoundError(f"No images found in {style_folder}")
    chosen = random.choice(files)
    return os.path.join(style_folder, chosen)


@torch.no_grad()
def generate(model, vae, style_img_path, text, out_path, img_size=(64, 1024), seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    style_img = Image.open(style_img_path).convert("RGB")
    style_img = transform(style_img).unsqueeze(0).to(device)

    # conditioning is fixed across all denoising steps -> compute once
    with autocast(device_type=device.type):
        ver_map, hor_map, _, _, _ = model.style_encoder(style_img)
        Q = model.content_encoder([text])
        cond = model.blender(Q, ver_map, hor_map)

    latent = torch.randn(1, 4, 8, 128, device=device)

    for t in reversed(range(0, TIMESTEPS)):
        t_tensor = torch.tensor([t], device=device)
        with autocast(device_type=device.type):
            time_emb = model.time_embedding(t_tensor)
            pred_noise = model.unet(latent, cond, time_emb)
            pred_noise = model.final(pred_noise)

        alpha, alpha_bar, beta = alphas[t], alpha_bars[t], betas[t]
        latent = (1 / torch.sqrt(alpha)) * (
            latent - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
        )
        if t > 0:
            latent = latent + torch.sqrt(beta) * torch.randn_like(latent)
        latent = torch.clamp(latent, -3.0, 3.0)

    latent = latent / 0.18215
    with autocast(device_type=device.type):
        img = vae.decode(latent).sample

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_image((img.clamp(-1, 1) + 1) / 2, out_path)
    print(f"[SAVED] {out_path}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for the DiffBrush-style handwriting model")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--style_folder", required=True, help="Folder with style images")
    parser.add_argument("--style_image", default=None, help="Specific style image path")
    parser.add_argument("--text", default=None, help="Text to render (English ITRANS or Devanagari)")
    parser.add_argument("--out_dir", default="./inference_out")
    parser.add_argument("--out_name", default=None, help="Output filename")
    parser.add_argument("--font_path", default="/home/kishan/diffusion/NotoSansDevanagari-Regular.ttf")
    parser.add_argument("--vae_path", default="/home/kishan/diffusion/vae")
    parser.add_argument("--img_height", type=int, default=64, help="Style image resize height")
    parser.add_argument("--img_width", type=int, default=1024, help="Style image resize width")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Get input interactively if not provided via CLI
    hindi_text = args.text
    if hindi_text is None:
        hindi_text = input("Please enter the text to render (English or Hindi): ")

    # --- TRANSLITERATION LOGIC ---
    # If the text is made of standard English/ASCII characters, convert it to Hindi
    if hindi_text.isascii():
        original = hindi_text
        hindi_text = transliterate(hindi_text, sanscript.ITRANS, sanscript.DEVANAGARI)
        print(f"[TEXT CONVERTED] '{original}' -> '{hindi_text}'")
    else:
        print(f"[TEXT PROVIDED] '{hindi_text}' (Native Devanagari detected)")
    # -----------------------------

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    model = load_model_for_inference(args.ckpt, args.font_path)

    style_img_path = pick_style_image(args.style_folder, args.style_image)
    print(f"[STYLE IMAGE] {style_img_path}")

    # Set up randomized filename
    if args.out_name is None:
        R = random.randint(1, 1000)
        print(f"Random number R generated: {R}")
        args.out_name = f"output_img_{R}.png"

    out_path = os.path.join(args.out_dir, args.out_name)

    generate(
        model, vae, style_img_path, hindi_text, out_path,
        img_size=(args.img_height, args.img_width),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
