"""
compute_dcer_dwer.py
================================
OCR-based text-recognition accuracy for generated handwriting:
    DWER   - (Word Error Rate) via jiwer, needs OCR
    DCER   - (Character Error Rate) via jiwer, needs OCR

Extracted from the WER/CER portion of compute_all_metrics.py — this script
only runs OCR + jiwer scoring, none of the FID/HWD/GS/IS image-distribution
metrics.

Expected folder layout (same as the FID/HWD/GS scripts, this is what
hwd.FolderDataset-style layouts look like — though this script doesn't
actually need FolderDataset, just the images + JSON on disk):

    dataset_root/
    ├── author1/
    │   ├── sample1.png
    │   ├── sample2.png
    │   └── ...
    ├── author2/
    │   └── ...
    └── transcriptions.json      <-- REQUIRED
                                      {"author1/sample1.png": "ground truth text", ...}

    NOTE: transcriptions.json keys don't have to match the on-disk layout
    exactly. If a key's path doesn't resolve under dataset_root (e.g. the
    JSON was written for a nested layout but images now live flat, or vice
    versa), the loader falls back to matching by basename against an index
    of every image actually found under dataset_root. If a filename can't
    be matched at all, it's skipped and reported — WER/CER only ever runs
    over images that were actually found and actually OCR'd.

-------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------
1. Requirements (install once):
       pip install jiwer torch transformers accelerate qwen-vl-utils tqdm pillow

2. Make sure `dataset_root` (the --data argument) contains:
       - the images (any nested folder layout)
       - a transcriptions.json at its top level, in the format shown above

3. Choose an OCR backend:
       --ocr qwen     -> zero-shot Qwen2.5-VL reader (default, no training
                          needed, good general-purpose choice for
                          Devanagari handwriting)
       --ocr parseq   -> slot for your own trained PARSeq recognizer; you
                          must fill in the TODO in PARSeqOCR.__init__/.read
                          and pass --parseq-ckpt pointing at your checkpoint

4. Run it:
       python compute_dcer_dwer.py --data /path/to/dataset_root --ocr qwen

   Optional flags:
       --parseq-ckpt PATH   required only when --ocr parseq

   Output:
       - A tqdm progress bar while each image is OCR'd
       - Warnings for any transcription entries that couldn't be matched
         to an image on disk, and if more than 10% of entries were
         skipped, a note that the score is over a reduced sample
       - Printed DWER and DCER to the console
-------------------------------------------------------------------------
"""

import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

import jiwer


# =============================================================================
# OCR backends (pluggable)
# =============================================================================

class OCRBackend:
    """Base interface — implement `read(PIL.Image) -> str`."""
    def read(self, image: Image.Image) -> str:
        raise NotImplementedError


class QwenVLOCR(OCRBackend):
    """
    Zero-shot handwriting reader using Qwen2.5-VL. Good general-purpose choice
    for Devanagari handwriting when you don't have a domain-tuned recognizer
    handy. Swap for your trained PARSeq model (see PARSeqOCR below) for best
    accuracy on your exact data distribution.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = None):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.prompt = (
            "Transcribe the handwritten Hindi (Devanagari) text in this image exactly as "
            "written. Output ONLY the transcribed text, no explanation, no translation."
        )

    @torch.no_grad()
    def read(self, image: Image.Image) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        result = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return result.strip()


class PARSeqOCR(OCRBackend):
    """
    Drop-in slot for your own trained PARSeq recognizer (the one you trained
    on synthetic + IIIT-INDIC-HW-Words). This will almost certainly beat any
    general VLM on your domain — wire up your checkpoint loading + inference
    here and pass --ocr parseq.
    """
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: load your PARSeq checkpoint here, e.g.:
        # from strhub.models.utils import load_from_checkpoint
        # self.model = load_from_checkpoint(checkpoint_path).eval().to(self.device)
        raise NotImplementedError(
            "Wire up your PARSeq checkpoint loading + decode logic here, "
            "then this becomes the highest-accuracy option for your data."
        )

    @torch.no_grad()
    def read(self, image: Image.Image) -> str:
        raise NotImplementedError


def build_ocr(backend: str, parseq_ckpt: str = None) -> OCRBackend:
    if backend == "qwen":
        return QwenVLOCR()
    if backend == "parseq":
        if not parseq_ckpt:
            raise ValueError("--parseq-ckpt is required when --ocr parseq")
        return PARSeqOCR(parseq_ckpt)
    raise ValueError(f"Unknown OCR backend: {backend}")


# =============================================================================
# DWER / DCER
# =============================================================================

def build_file_index(dataset_root: str) -> dict:
    """
    Map basename -> full Path for every image actually found under
    dataset_root. Used as a fallback when a transcriptions.json key doesn't
    resolve directly (e.g. JSON expects a nested layout but images are flat,
    or vice versa).
    """
    index = {}
    dupes = set()
    for p in Path(dataset_root).rglob("*"):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            if p.name in index and p.name not in dupes:
                dupes.add(p.name)
            index[p.name] = p
    if dupes:
        print(f"  [warn] {len(dupes)} duplicate filename(s) on disk under {dataset_root} — "
              f"basename fallback matching is ambiguous for these, last one found wins")
    return index


def resolve_image_path(dataset_root: str, rel_path: str, file_index: dict):
    """
    Try the literal join first (handles the normal, correctly-nested case).
    If that doesn't exist on disk, fall back to matching by basename against
    the file_index. Returns None if neither resolves.
    """
    direct = Path(dataset_root) / rel_path
    if direct.exists():
        return direct

    # os.path.join / Path(a) / b will silently drop `a` if `rel_path` is
    # absolute — guard against that explicitly too.
    if Path(rel_path).is_absolute() and Path(rel_path).exists():
        return Path(rel_path)

    fallback = file_index.get(os.path.basename(rel_path))
    return fallback


def run_ocr_and_score(dataset_root: str, ocr: OCRBackend):
    transcriptions_path = os.path.join(dataset_root, "transcriptions.json")
    if not os.path.exists(transcriptions_path):
        raise FileNotFoundError(
            f"transcriptions.json not found at {transcriptions_path}. "
            "DWER/DCER need ground-truth text per image (see docstring for the format)."
        )
    with open(transcriptions_path, "r", encoding="utf-8") as f:
        transcriptions = json.load(f)

    if not transcriptions:
        raise RuntimeError(f"transcriptions.json at {transcriptions_path} is empty.")

    file_index = build_file_index(dataset_root)

    preds, gts, missing = [], [], []
    for rel_path, gt_text in tqdm(transcriptions.items(), desc="OCR inference (DWER/DCER)"):
        img_path = resolve_image_path(dataset_root, rel_path, file_index)
        if img_path is None:
            missing.append(rel_path)
            continue

        image = Image.open(img_path).convert("RGB")
        pred_text = ocr.read(image)
        preds.append(pred_text)
        gts.append(gt_text)

    if missing:
        print(f"  [warn] {len(missing)}/{len(transcriptions)} images could not be matched "
              f"on disk (even with basename fallback) under {dataset_root}")
        for m in missing[:5]:
            print(f"    e.g. missing: {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    if not preds:
        raise RuntimeError(
            f"0/{len(transcriptions)} transcriptions matched any image under {dataset_root}. "
            "DWER/DCER cannot be computed. Check that transcriptions.json keys correspond "
            "to files that actually exist under --data, either by relative path or basename."
        )

    skip_rate = 1 - len(preds) / len(transcriptions)
    if skip_rate > 0.1:
        print(f"  [warn] {skip_rate:.1%} of transcriptions were skipped — "
              f"DWER/DCER below is computed over only {len(preds)}/{len(transcriptions)} "
              f"samples and may not be representative")

    wer = jiwer.wer(gts, preds)
    cer = jiwer.cer(gts, preds)
    return wer, cer, list(zip(gts, preds))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute DWER and DCER via OCR")
    parser.add_argument("--data", required=True,
                         help="Path to dataset root (images + transcriptions.json)")
    parser.add_argument("--ocr", default="qwen", choices=["qwen", "parseq"])
    parser.add_argument("--parseq-ckpt", default=None)
    args = parser.parse_args()

    ocr = build_ocr(args.ocr, args.parseq_ckpt)
    wer, cer, pairs = run_ocr_and_score(args.data, ocr)

    print("\n===== Summary =====")
    print(f"DWER: {wer:.4f}")
    print(f"DCER: {cer:.4f}")

    return wer, cer, pairs


if __name__ == "__main__":
    main()
