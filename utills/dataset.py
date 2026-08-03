import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HandwritingProxyNCALatentDataset(Dataset):
    def __init__(self, data_root_pt, data_root_jpg, image_size=(64, 1024), latent_ext=".pt"):
        self.data_root_pt = data_root_pt
        self.data_root_jpg = data_root_jpg
        self.latent_ext = latent_ext
        self.samples = []
        self.writer_to_id = {}

        self.img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        writers = sorted(os.listdir(data_root_pt))
        writer_idx = 0

        for writer in writers:
            writer_path_pt = os.path.join(data_root_pt, writer)
            writer_path_jpg = os.path.join(data_root_jpg, writer)

            if not os.path.isdir(writer_path_pt):
                continue

            label_path = os.path.join(writer_path_pt, "labels.txt")
            if not os.path.exists(label_path):
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]

            latent_files = sorted(
                [f for f in os.listdir(writer_path_pt) if f.endswith(self.latent_ext)],
                key=lambda x: int(os.path.splitext(x)[0]),
            )

            if not latent_files or not texts:
                continue

            if len(latent_files) != len(texts):
                print(f"[WARNING] Skipping {writer}: latents ({len(latent_files)}) vs labels ({len(texts)}) mismatch")
                continue

            self.writer_to_id[writer] = writer_idx
            writer_idx += 1

            for latent_name, txt in zip(latent_files, texts):
                base_name = os.path.splitext(latent_name)[0]
                jpg_name = f"{base_name}.jpg"
                jpg_path = os.path.join(writer_path_jpg, jpg_name)

                # Fallback to png if jpg not present
                if not os.path.exists(jpg_path):
                    jpg_path = os.path.join(writer_path_jpg, f"{base_name}.png")

                self.samples.append({
                    "latent_path": os.path.join(writer_path_pt, latent_name),
                    "image_path": jpg_path,
                    "text": txt,
                    "writer_id": self.writer_to_id[writer],
                })

        print(f"[Dataset] Total writers: {len(self.writer_to_id)}")
        print(f"[Dataset] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load Latent Tensor
        if self.latent_ext == ".pt":
            latent = torch.load(sample["latent_path"], weights_only=True)
        elif self.latent_ext == ".npy":
            latent = torch.from_numpy(np.load(sample["latent_path"]))
        else:
            raise ValueError(f"Unsupported latent extension: {self.latent_ext}")

        # 2. Load Style JPG Image
        if os.path.exists(sample["image_path"]):
            img = Image.open(sample["image_path"]).convert("RGB")
        else:
            # Fallback dummy white image if missing
            img = Image.new("RGB", (1024, 64), (255, 255, 255))
        
        img_tensor = self.img_transform(img)

        return latent.float(), img_tensor, sample["text"], sample["writer_id"]
