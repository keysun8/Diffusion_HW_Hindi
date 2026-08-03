import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models.diffusion import Diffusion, ProxyNCALoss
from utils.dataset import HandwritingProxyNCALatentDataset
from utils.diffusion_utils import get_diffusion_schedules, forward_diffusion, ddim_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiffBrush-Hindi Diffusion Model")
    parser.add_argument("--data_root_pt", type=str, required=True, help="Directory containing .pt latent files")
    parser.add_argument("--data_root_jpg", type=str, required=True, help="Directory containing .jpg image files")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples", help="Directory to save preview sample images")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint and samples every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num workers")
    parser.add_argument("--font_path", type=str, default="NotoSansDevanagari-Regular.ttf", help="Path to Devanagari font")
    parser.add_argument("--vae_path", type=str, required=True,
                         help="Path or HF repo id of the VAE used to encode your latents (must match "
                              "the VAE used during latent pre-encoding, NOT a generic stock VAE)")
    return parser.parse_args()


def save_checkpoint(epoch, model, proxy_losses, opt_style, opt_diff, scaler_style, scaler_diff, history, path):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "style_encoder": model.style_encoder.state_dict(),
        "content_encoder": model.content_encoder.state_dict(),
        "blender": model.blender.state_dict(),
        "unet": model.unet.state_dict(),
        "final": model.final.state_dict(),
        "embedding_head": model.embedding_head.state_dict(),
        "ver_proxy": proxy_losses["ver"].state_dict(),
        "hor_proxy": proxy_losses["hor"].state_dict(),
        "global_proxy": proxy_losses["global"].state_dict(),
        "opt_style": opt_style.state_dict(),
        "opt_diff": opt_diff.state_dict(),
        "scaler_style": scaler_style.state_dict(),
        "scaler_diff": scaler_diff.state_dict(),
        "loss_history": history,
    }
    ckpt_file = os.path.join(path, f"ckpt_epoch_{epoch}.pt")
    torch.save(ckpt, ckpt_file)
    print(f"[CHECKPOINT] Saved: {ckpt_file}")


def plot_losses(loss_history, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("diffusion_mse", axes[0, 0], "Diffusion MSE"),
        ("style_ver_nca", axes[0, 1], "Vertical Proxy-NCA"),
        ("style_hor_nca", axes[1, 0], "Horizontal Proxy-NCA"),
        ("global_nca", axes[1, 1], "Global Proxy-NCA"),
    ]
    for key, ax, title in metrics:
        if loss_history.get(key):
            ax.plot(loss_history[key])
            ax.set_title(title)
            ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_and_save_samples(model, sample_batch, epoch, sample_dir, device, vae_path):
    os.makedirs(sample_dir, exist_ok=True)
    model.eval()

    _, style_imgs, texts, _ = sample_batch
    style_imgs = style_imgs[:2].to(device)
    sample_texts = list(texts[:2])

    with torch.no_grad():
        sampled_latents = ddim_sample(
            model=model,
            style_img=style_imgs,
            text=sample_texts,
            shape=(len(sample_texts), 4, 8, 128),
            device=device,
            ddim_steps=20
        )

        try:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
            decoded = vae.decode(sampled_latents / 0.18215).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)

            save_tf = transforms.ToPILImage()
            for i in range(decoded.shape[0]):
                img = save_tf(decoded[i].cpu())
                out_path = os.path.join(sample_dir, f"epoch_{epoch}_sample_{i}.png")
                img.save(out_path)
            print(f"[SAMPLES] Rendered sample images saved to '{sample_dir}/'")
        except Exception:
            for i in range(sampled_latents.shape[0]):
                latent_vis = sampled_latents[i, :3].cpu().clamp(-1, 1)
                latent_vis = (latent_vis + 1) / 2
                save_tf = transforms.ToPILImage()
                img = save_tf(latent_vis)
                out_path = os.path.join(sample_dir, f"epoch_{epoch}_latent_{i}.png")
                img.save(out_path)
            print(f"[SAMPLES] Saved latent previews to '{sample_dir}/'")


def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    print("--- Launching Training ---")
    print(f"Data Root Latents (.pt): {args.data_root_pt}")
    print(f"Data Root Images  (.jpg): {args.data_root_jpg}")
    print(f"Checkpoint Directory:    {args.ckpt_dir}")
    print(f"Sample Directory:        {args.sample_dir}")
    print(f"Epochs:                  {args.epochs}")
    print(f"Batch Size:              {args.batch_size}")
    print(f"Device:                  {device}\n")

    dataset = HandwritingProxyNCALatentDataset(
        data_root_pt=args.data_root_pt,
        data_root_jpg=args.data_root_jpg
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    num_writers = len(dataset.writer_to_id)
    model = Diffusion(font_path=args.font_path).to(device)

    proxy_losses = {
        "ver": ProxyNCALoss(num_writers, 512).to(device),
        "hor": ProxyNCALoss(num_writers, 512).to(device),
        "global": ProxyNCALoss(num_writers, 512).to(device),
    }

    opt_style = torch.optim.AdamW(
        list(model.style_encoder.parameters()) +
        list(proxy_losses["ver"].parameters()) +
        list(proxy_losses["hor"].parameters()),
        lr=args.lr
    )

    opt_diff = torch.optim.AdamW(
        list(model.content_encoder.parameters()) +
        list(model.blender.parameters()) +
        list(model.unet.parameters()) +
        list(model.final.parameters()) +
        list(model.embedding_head.parameters()) +
        list(proxy_losses["global"].parameters()),
        lr=args.lr
    )

    scaler_style = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    scaler_diff = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    _, _, _, sqrt_ab, sqrt_one_minus_ab = get_diffusion_schedules(device)
    loss_history = {"diffusion_mse": [], "style_ver_nca": [], "style_hor_nca": [], "global_nca": []}

    fixed_sample_batch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_mse, total_v, total_h, total_g = 0.0, 0.0, 0.0, 0.0

        for batch_idx, (latents, style_imgs, texts, writer_ids) in enumerate(dataloader):
            if fixed_sample_batch is None:
                fixed_sample_batch = (latents, style_imgs, texts, writer_ids)

            latents = latents.to(device)
            style_imgs = style_imgs.to(device)
            writer_ids = writer_ids.to(device)

            b = latents.shape[0]
            t = torch.randint(0, 1000, (b,), device=device).long()

            noisy_latents, noise = forward_diffusion(latents, t, sqrt_ab, sqrt_one_minus_ab)

            # 1. Style Step
            opt_style.zero_grad()
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                _, _, ver_emb, hor_emb, _ = model.style_encoder(style_imgs)
                loss_v = proxy_losses["ver"](ver_emb, writer_ids)
                loss_h = proxy_losses["hor"](hor_emb, writer_ids)
                loss_style = loss_v + loss_h

            scaler_style.scale(loss_style).backward()
            scaler_style.step(opt_style)
            scaler_style.update()

            # 2. Diffusion Step
            opt_diff.zero_grad()
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                pred_noise, glob_emb, _, _, _ = model(noisy_latents, style_imgs, texts, t)
                loss_mse = torch.nn.functional.mse_loss(pred_noise, noise)
                loss_g = proxy_losses["global"](glob_emb, writer_ids)
                loss_diff = loss_mse + 0.1 * loss_g

            scaler_diff.scale(loss_diff).backward()
            scaler_diff.step(opt_diff)
            scaler_diff.update()

            total_mse += loss_mse.item()
            total_v += loss_v.item()
            total_h += loss_h.item()
            total_g += loss_g.item()

        num_batches = len(dataloader)
        avg_mse = total_mse / num_batches
        avg_v = total_v / num_batches
        avg_h = total_h / num_batches
        avg_g = total_g / num_batches

        loss_history["diffusion_mse"].append(avg_mse)
        loss_history["style_ver_nca"].append(avg_v)
        loss_history["style_hor_nca"].append(avg_h)
        loss_history["global_nca"].append(avg_g)

        print(f"Epoch [{epoch}/{args.epochs}] | MSE Loss: {avg_mse:.4f} | Style (V/H): {avg_v:.3f}/{avg_h:.3f} | Global: {avg_g:.3f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(epoch, model, proxy_losses, opt_style, opt_diff, scaler_style, scaler_diff, loss_history, args.ckpt_dir)
            plot_losses(loss_history, os.path.join(args.sample_dir, "loss_curves.png"))
            if fixed_sample_batch is not None:
                generate_and_save_samples(model, fixed_sample_batch, epoch, args.sample_dir, device, args.vae_path)


if __name__ == "__main__":
    train()
