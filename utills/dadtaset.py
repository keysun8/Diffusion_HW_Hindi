import os
import torch
import numpy as np
from torch.utils.data import Dataset

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
                print(f"[WARNING] Skipping {writer}: latents ({len(latent_files)}) vs labels ({len(texts)}) mismatch")
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
        return latent.float(), sample["text"], sample["writer_id"]
