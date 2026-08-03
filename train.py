# Dataset 

class HandwritingProxyNCALatentDataset(Dataset):
    def __init__(self, root_dir, latent_ext=".pt"):
        self.samples = []
        self.writer_to_id = {}
        self.latent_ext = latent_ext

        writers = sorted(os.listdir(root_dir))
        writer_idx = 0

        for writer in writers:
            writer_path = os.path.join(root_dir, writer)
            if not os.path.isdir(writer_path):
                continue
            label_path = os.path.join(writer_path, "labels.txt")
            if not os.path.exists(label_path):
                continue
            with open(label_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]

            latent_files = sorted(
                [f for f in os.listdir(writer_path) if f.endswith(self.latent_ext)],
                key=lambda x: int(os.path.splitext(x)[0]),
            )
            if not latent_files or not texts:
                continue
            if len(latent_files) != len(texts):
                print(f"[WARNING] skipping {writer}: latents ({len(latent_files)}) "
                      f"and labels ({len(texts)}) mismatch")
                continue

            self.writer_to_id[writer] = writer_idx
            writer_idx += 1

            for latent_name, txt in zip(latent_files, texts):
                self.samples.append({
                    "latent_path": os.path.join(writer_path, latent_name),
                    "text": txt,
                    "writer_id": self.writer_to_id[writer],
                })

        print(f"Total writers: {len(self.writer_to_id)}")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.latent_ext == ".pt":
            latent = torch.load(sample["latent_path"], weights_only=True)
        elif self.latent_ext == ".npy":
            latent = torch.from_numpy(np.load(sample["latent_path"]))
        else:
            raise ValueError(f"Unsupported latent extension: {self.latent_ext}")
        latent = latent.float()
        return latent, sample["text"], sample["writer_id"]


# Checkpointing (fixed key mismatch between save/load)

def save_checkpoint(epoch, model_diffusion, proxy_losses, optimizer_style,
                     optimizer_diffusion, scaler_style, scaler_diffusion,
                     loss_history, path="diff_checkpoints_5"):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "style_encoder": model_diffusion.style_encoder.state_dict(),
        "content_encoder": model_diffusion.content_encoder.state_dict(),
        "blender": model_diffusion.blender.state_dict(),
        "unet": model_diffusion.unet.state_dict(),
        "final": model_diffusion.final.state_dict(),
        "embedding_head": model_diffusion.embedding_head.state_dict(),
        "ver_proxy_loss": proxy_losses["ver"].state_dict(),
        "hor_proxy_loss": proxy_losses["hor"].state_dict(),
        "global_proxy_loss": proxy_losses["global"].state_dict(),
        "optimizer_style": optimizer_style.state_dict(),
        "optimizer_diffusion": optimizer_diffusion.state_dict(),
        "scaler_style": scaler_style.state_dict(),
        "scaler_diffusion": scaler_diffusion.state_dict(),
        "loss_history": loss_history,
    }
    torch.save(ckpt, f"{path}/ckpt_epoch_{epoch}.pt")
    print(f"[CHECKPOINT SAVED] epoch {epoch}")


def load_checkpoint(path, model_diffusion, proxy_losses, optimizer_style,
                     optimizer_diffusion, scaler_style, scaler_diffusion):
    ckpt = torch.load(path, map_location=device)
    model_diffusion.style_encoder.load_state_dict(ckpt["style_encoder"])
    model_diffusion.content_encoder.load_state_dict(ckpt["content_encoder"])
    model_diffusion.blender.load_state_dict(ckpt["blender"])
    model_diffusion.unet.load_state_dict(ckpt["unet"])
    model_diffusion.final.load_state_dict(ckpt["final"])
    model_diffusion.embedding_head.load_state_dict(ckpt["embedding_head"])
    proxy_losses["ver"].load_state_dict(ckpt["ver_proxy_loss"])
    proxy_losses["hor"].load_state_dict(ckpt["hor_proxy_loss"])
    proxy_losses["global"].load_state_dict(ckpt["global_proxy_loss"])
    optimizer_style.load_state_dict(ckpt["optimizer_style"])
    optimizer_diffusion.load_state_dict(ckpt["optimizer_diffusion"])
    scaler_style.load_state_dict(ckpt["scaler_style"])
    scaler_diffusion.load_state_dict(ckpt["scaler_diffusion"])
    print(f"[CHECKPOINT LOADED] epoch {ckpt['epoch']}")
    return ckpt["epoch"], ckpt.get("loss_history", {})


# Loss-curve plotting

def plot_losses(loss_history, out_path="loss_curves.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    keys_axes = [
        ("diffusion_mse", axes[0, 0], "Diffusion MSE"),
        ("style_ver_nca", axes[0, 1], "Vertical Proxy-NCA"),
        ("style_hor_nca", axes[1, 0], "Horizontal Proxy-NCA"),
        ("global_nca", axes[1, 1], "Global Proxy-NCA"),
    ]
    for key, ax, title in keys_axes:
        if loss_history.get(key):
            ax.plot(loss_history[key])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# Image generation each epoch -- also returns generated tensors for FID

@torch.no_grad()
def generate_images(model, vae, style_folder, texts, device, epoch, step=None, out_dir="eval_generated_v5"):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((64, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
    style_imgs_files = sorted(
        f for f in os.listdir(style_folder)
        if f.lower().endswith(IMG_EXTS)
    )[:len(texts)]

    if len(style_imgs_files) < len(texts):                      # <-- ADD THIS
        print(f"[WARN] only {len(style_imgs_files)} images in {style_folder}, "
              f"fewer than {len(texts)} requested test_texts")

    os.makedirs(out_dir, exist_ok=True)
    
    generated, reals = [], []

    for i, (img_name, text) in enumerate(zip(style_imgs_files, texts)):
        img_path = os.path.join(style_folder, img_name)
        style_img_pil = Image.open(img_path).convert("RGB")
        style_img = transform(style_img_pil).unsqueeze(0).to(device)
        reals.append(style_img.squeeze(0).cpu())

        # --- conditioning doesn't depend on t: compute it ONCE ---
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
        generated.append(img.squeeze(0).cpu())

        style_name = os.path.basename(os.path.normpath(style_folder))

        if step is not None:
            save_image((img.clamp(-1, 1) + 1) / 2,
                        f"{out_dir}/epoch_{epoch}_step_{step}_style_{style_name}_{i}.png")
        else:
            save_image((img.clamp(-1, 1) + 1) / 2,
                        f"{out_dir}/epoch_{epoch}_style_{style_name}_{i}.png")

    model.train()
    return reals, generated

def pick_random_style_folder(base_dir, folder_range):
    lo, hi = folder_range
    idx = random.randint(lo, hi)
    return os.path.join(base_dir, str(idx))


def train_2(model_diffusion, dataset, loader, vae, num_epochs=10,
            lambda_ver_nca=0.8, lambda_hor_nca=0.8, lambda_global_nca=0,
            checkpoint_every=5, resume_from=None,
            style_base_dir="/home/kishan/diffusion/output_dataset_hindi_with_json_line/test",
            style_folder_range=(1, 75),
            fid_style_folder="/home/kishan/diffusion/output_dataset_hindi_with_json_line/test/2",
            test_texts=(
                "नमस्ते धन्यवाद स्वागत", "मैं किशन हूँ।", "भाड़ में जाओ",
                "मैं अभी भी ड्राइवर हूँ।", "आज मौसम अच्छा है", "मुझे भूख लगी है",
            )):

    mse_fn = nn.MSELoss()
    proxy_losses = {
        "ver": ProxyNCALoss(num_classes=len(dataset.writer_to_id), embedding_dim=512).to(device),
        "hor": ProxyNCALoss(num_classes=len(dataset.writer_to_id), embedding_dim=512).to(device),
        "global": ProxyNCALoss(num_classes=len(dataset.writer_to_id), embedding_dim=512).to(device),
    }

    optimizer_style = torch.optim.AdamW(
        list(model_diffusion.style_encoder.parameters())
        + list(proxy_losses["ver"].parameters())
        + list(proxy_losses["hor"].parameters()),
        lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.99),
    )

    
    optimizer_diffusion = torch.optim.AdamW([
        {"params": model_diffusion.content_encoder.parameters(), "lr": 5e-5},
        {"params": model_diffusion.blender.parameters(), "lr": 5e-5},
        {"params": model_diffusion.unet.parameters(), "lr": 1e-5},
        {"params": model_diffusion.final.parameters(), "lr": 1e-5},
        {"params": model_diffusion.embedding_head.parameters(), "lr": 1e-5},
        {"params": proxy_losses["global"].parameters(), "lr": 1e-5},
    ], weight_decay=1e-4, betas=(0.9, 0.99))

    scaler_style = GradScaler()
    scaler_diffusion = GradScaler()

    loss_history = {"diffusion_mse": [], "style_ver_nca": [], "style_hor_nca": [], "global_nca": [], "fid": []}
    start_epoch = 0

    if resume_from is not None and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model_diffusion.style_encoder.load_state_dict(ckpt["style_encoder"])
        model_diffusion.content_encoder.load_state_dict(ckpt["content_encoder"])
        model_diffusion.blender.load_state_dict(ckpt["blender"])
        model_diffusion.unet.load_state_dict(ckpt["unet"])
        model_diffusion.final.load_state_dict(ckpt["final"])
        model_diffusion.embedding_head.load_state_dict(ckpt["embedding_head"])
        proxy_losses["ver"].load_state_dict(ckpt["ver_proxy_loss"])
        proxy_losses["hor"].load_state_dict(ckpt["hor_proxy_loss"])
        proxy_losses["global"].load_state_dict(ckpt["global_proxy_loss"])
        optimizer_style.load_state_dict(ckpt["optimizer_style"])
        scaler_style.load_state_dict(ckpt["scaler_style"])
        scaler_diffusion.load_state_dict(ckpt["scaler_diffusion"])

        start_epoch = ckpt["epoch"] + 1
        loss_history.update(ckpt.get("loss_history", {}))

        completed = len(loss_history.get("diffusion_mse", []))
        print(f"[RESUME] checkpoint epoch {ckpt['epoch']} loaded "
              f"({completed} epochs of history) | "
              f"resuming at epoch {start_epoch}/{num_epochs} "
              f"({num_epochs - start_epoch} epochs remaining) | "
              f"optimizer_diffusion REINITIALIZED with LR split "
              f"(content_encoder/blender=5e-5, unet/final/embedding_head=1e-5)")

        if completed != start_epoch:
            print(f"[RESUME][WARNING] loss_history length ({completed}) "
                  f"doesn't match resumed epoch index ({start_epoch}) — "
                  f"curves may be misaligned if you changed num_epochs "
                  f"or skipped a save between runs.")

        if start_epoch >= num_epochs:
            print(f"[RESUME] start_epoch ({start_epoch}) >= num_epochs "
                  f"({num_epochs}) — nothing to train, increase num_epochs.")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    for epoch in range(start_epoch, num_epochs):
        model_diffusion.train()
        pbar = tqdm(loader)

        running_mse, running_ver, running_hor, running_global, n_batches = 0.0, 0.0, 0.0, 0.0, 0

        for latents, texts, writer_ids in pbar:
            latents = latents.to(device)
            writer_ids = writer_ids.to(device)
            B = latents.shape[0]

            target_latents = latents
            with torch.no_grad():
                style_imgs = vae.decode(latents / 0.18215).sample

            t = torch.randint(0, TIMESTEPS, (B,), device=device).long()
            xt, noise = forward_diffusion(target_latents, t)

           
            optimizer_style.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                _, _, ver_emb, hor_emb, _ = model_diffusion.style_encoder(style_imgs)

                if torch.isnan(ver_emb).any() or torch.isnan(hor_emb).any():
                    print("NaN in ver_emb/hor_emb, skipping batch")
                    continue

                ver_emb = torch.clamp(ver_emb, -10, 10)
                hor_emb = torch.clamp(hor_emb, -10, 10)

                style_ver_loss = proxy_losses["ver"](ver_emb, writer_ids)
                style_hor_loss = proxy_losses["hor"](hor_emb, writer_ids)
                style_loss_total = lambda_ver_nca * style_ver_loss + lambda_hor_nca * style_hor_loss

            if torch.isnan(style_loss_total):
                print("NaN style_loss_total, skipping batch")
                continue

            scaler_style.scale(style_loss_total).backward()
            scaler_style.unscale_(optimizer_style)
            torch.nn.utils.clip_grad_norm_(model_diffusion.style_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proxy_losses["ver"].parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proxy_losses["hor"].parameters(), 1.0)
            scaler_style.step(optimizer_style)
            scaler_style.update()

           
            optimizer_diffusion.zero_grad(set_to_none=True)

            with torch.no_grad():
                ver_map, hor_map, _, _, _ = model_diffusion.style_encoder(style_imgs)

            with autocast(device_type=device.type):
                Q = model_diffusion.content_encoder(texts)
                cond = model_diffusion.blender(Q, ver_map, hor_map)
                time_emb = model_diffusion.time_embedding(t)
                pred_noise = model_diffusion.unet(xt, cond, time_emb)
                pred_noise = model_diffusion.final(pred_noise)
                global_emb = model_diffusion.embedding_head(pred_noise.detach())

                if torch.isnan(pred_noise).any() or torch.isnan(global_emb).any():
                    print("NaN in pred_noise/global_emb, skipping batch")
                    continue

                global_emb = torch.clamp(global_emb, -10, 10)
                diffusion_loss = mse_fn(pred_noise.float(), noise.float())
                global_nca_loss = proxy_losses["global"](global_emb, writer_ids)
                diffusion_total_loss = diffusion_loss + lambda_global_nca * global_nca_loss

            if torch.isnan(diffusion_total_loss):
                print("NaN diffusion_total_loss, skipping batch")
                continue

            scaler_diffusion.scale(diffusion_total_loss).backward()
            scaler_diffusion.unscale_(optimizer_diffusion)
            for module in (model_diffusion.content_encoder, model_diffusion.blender,
                           model_diffusion.unet, model_diffusion.final, model_diffusion.embedding_head):
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            scaler_diffusion.step(optimizer_diffusion)
            scaler_diffusion.update()

            running_mse += diffusion_loss.item()
            running_ver += style_ver_loss.item()
            running_hor += style_hor_loss.item()
            running_global += global_nca_loss.item()
            n_batches += 1

            pbar.set_description(
                f"Epoch {epoch+1} | Diff: {diffusion_loss.item():.4f} | "
                f"Ver: {style_ver_loss.item():.4f} | Hor: {style_hor_loss.item():.4f} | "
                f"Global: {global_nca_loss.item():.4f}"
            )

            if n_batches % 500 == 0:
                preview_style_folder = pick_random_style_folder(style_base_dir, style_folder_range)
                print(f"\n[Mid-epoch Generation] Epoch {epoch}, Step {n_batches}... "
                      f"using style folder: {preview_style_folder}")
                generate_images(
                    model_diffusion, vae, style_folder=preview_style_folder,
                    texts=list(test_texts), device=device, epoch=epoch, step=n_batches
                )

        n_batches = max(n_batches, 1)
        loss_history["diffusion_mse"].append(running_mse / n_batches)
        loss_history["style_ver_nca"].append(running_ver / n_batches)
        loss_history["style_hor_nca"].append(running_hor / n_batches)
        loss_history["global_nca"].append(running_global / n_batches)

        preview_style_folder = pick_random_style_folder(style_base_dir, style_folder_range)
        print(f"[Epoch Generation] preview style folder: {preview_style_folder}")
        generate_images(
            model_diffusion, vae, style_folder=preview_style_folder,
            texts=list(test_texts), device=device, epoch=epoch,
        )

        reals, fakes = generate_images(
            model_diffusion, vae, style_folder=fid_style_folder,
            texts=list(test_texts), device=device, epoch=epoch,
        )
        try:
            fid_score = compute_fid(reals, fakes, device)
        except Exception as e:
            print(f"[FID] skipped this epoch ({e})")
            fid_score = float("nan")
        loss_history["fid"].append(fid_score)
        print(f"[EPOCH {epoch}] FID: {fid_score:.3f}")

        plot_losses(loss_history)
        plot_fid(loss_history)

        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                epoch, model_diffusion, proxy_losses, optimizer_style,
                optimizer_diffusion, scaler_style, scaler_diffusion, loss_history,
            )

    return loss_history

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params     : {total_params:,}")
    print(f"Trainable Params : {trainable_params:,}")
    return total_params, trainable_params


if __name__ == "__main__":
    dataset = HandwritingProxyNCALatentDataset(
        "/home/kishan/diffusion/output_dataset_hindi_with_json_line_latent/train",
        latent_ext=".pt",
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    vae = AutoencoderKL.from_pretrained("/home/kishan/diffusion/vae").to(device)
    model_diffusion = Diffusion().to(device)

    train_2(model_diffusion, dataset, loader, vae, num_epochs=800, checkpoint_every=5)
