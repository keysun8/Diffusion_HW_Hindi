"""
bulk_img_generation_2k_dcer.py
Generates images using the trained Diffusion model (TEST-style + TRAIN-style
folders), and records ground-truth text per image in transcriptions.json for
later DCER/DWER computation.

Text selection: sequential (first N lines from TEXT_LIST_PATH, in file order),
NOT random. Each pool (test/train) has its own cursor starting at line 0 of
the text file, so e.g. TRAIN will consume the first TRAIN_TARGET_TOTAL lines
in order. If you want TRAIN to continue where TEST left off instead of also
starting at line 0, pass the same cursor dict to both calls in main().

tqdm: updates per-image (not per-folder-batch), so the bar advances smoothly
as each image is generated instead of jumping in chunks of `per_folder`.

-------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------
1. Requirements (install once):
       pip install torch tqdm
   (plus whatever `updated_diff_brush` itself needs, e.g. diffusers,
   torchvision, etc. — install those the same way if missing.)

2. Make sure `updated_diff_brush.py` is importable, i.e. either:
       - it sits in the same folder as this script, or
       - its folder is on your PYTHONPATH.

3. Update the path constants near the top of this file so they point to
   your own machine's files/folders:
       CKPT_PATH        -> trained model checkpoint (.pt)
       VAE_PATH         -> folder with the pretrained VAE
       FONT_PATH        -> .ttf font file used for rendering
       TEST_STYLE_ROOT  -> folder containing TEST style subfolders
       TRAIN_STYLE_ROOT -> folder containing TRAIN style subfolders
       TEXT_LIST_PATH   -> .txt file with one text line per candidate string
       OUT_DIR          -> where generated images + transcriptions.json go

4. (Optional) Adjust the generation-size knobs:
       TEST_PER_FOLDER / TEST_TARGET_TOTAL
       TRAIN_PER_FOLDER / TRAIN_TARGET_TOTAL
       SKIP_ROWS (inside main()) -> how many lines of TEXT_LIST_PATH to
                                    skip before starting to pick text

5. Run it:
       python bulk_img_generation_2k_dcer.py

   Output:
       - Generated PNGs saved directly into OUT_DIR, named
         epoch_{test|train}_style_{style_name}_{i}.png
       - OUT_DIR/transcriptions.json mapping each image's relative path
         (tag/style_name/filename) to its ground-truth text, for later
         DCER/DWER scoring.
-------------------------------------------------------------------------
"""

import os
import json
import random
import torch
from pathlib import Path
from tqdm import tqdm

from updated_diff_brush import (
    Diffusion, AutoencoderKL, generate_images, device
)

print("Hello start !!")

CKPT_PATH = "/home/kishan/diffusion/diff_checkpoints_5/ckpt_epoch_599.pt"   
VAE_PATH = "/home/kishan/diffusion/vae"
FONT_PATH = "/home/kishan/diffusion/NotoSansDevanagari-Regular.ttf"

TEST_STYLE_ROOT = "/home/kishan/diffusion/Syn_Data/hw_restructured/test"     
TRAIN_STYLE_ROOT = "/home/kishan/diffusion/Syn_Data/hw_restructured/train"   
TEXT_LIST_PATH = "/home/kishan/diffusion/Syn_Data/hindi_script_3words_150k.txt"

OUT_DIR = "/home/kishan/diffusion/Syn_Data/hw_syn_3"

TEST_PER_FOLDER = 0       # -> ~525 across 75 folders, capped to 500
TEST_TARGET_TOTAL = 1

TRAIN_PER_FOLDER = 1000     # -> ~9000 across 900 folders, capped to TRAIN_TARGET_TOTAL
TRAIN_TARGET_TOTAL = 7000

random.seed(42)
os.makedirs(OUT_DIR, exist_ok=True)


def load_checkpoint_for_inference(path, model):
    ckpt = torch.load(path, map_location=device)
    model.style_encoder.load_state_dict(ckpt["style_encoder"])
    model.content_encoder.load_state_dict(ckpt["content_encoder"])
    model.blender.load_state_dict(ckpt["blender"])
    model.unet.load_state_dict(ckpt["unet"])
    model.final.load_state_dict(ckpt["final"])
    model.embedding_head.load_state_dict(ckpt["embedding_head"])
    print(f"[CHECKPOINT LOADED] epoch {ckpt.get('epoch', '?')}")
    return ckpt.get("epoch", 0)


def list_style_folders(root):
    return [f for f in Path(root).iterdir() if f.is_dir()]


def pick_folders_and_counts(root, per_folder, target_total):
    folders = list_style_folders(root)
    random.shuffle(folders)

    plan = []
    running_total = 0
    for folder in folders:
        n = per_folder
        if running_total + n > target_total:
            n = target_total - running_total
        if n <= 0:
            break
        plan.append((folder, n))
        running_total += n
        if running_total >= target_total:
            break

    if running_total < target_total:
        print(f"[WARN] Only planned {running_total}/{target_total} images from {root} "
              f"(not enough folders/images). Consider raising per_folder count.")

    return plan


def next_texts(all_texts, cursor, n):
    """
    Pull the next n texts sequentially (in file order) starting from
    cursor['pos']. Wraps around to the start if the list is exhausted
    (with a warning) rather than repeating randomly.
    """
    n_texts = len(all_texts)
    out = []
    for _ in range(n):
        if cursor["pos"] >= n_texts:
            print(f"[WARN] Ran out of texts, wrapping around from start "
                  f"(consumed {cursor['pos']} texts).")
            cursor["pos"] = 0
        out.append(all_texts[cursor["pos"]])
        cursor["pos"] += 1
    return out


def run_generation_for_pool(plan, all_texts, model, vae, tag, out_dir, desc,
                             transcriptions, cursor):
    """
    tag: 'test' or 'train' (used as the `epoch` arg to generate_images, and
         as the save-path prefix, matching generate_images' filename format:
         epoch_{tag}_style_{style_name}_{i}.png
    transcriptions: dict to populate with rel_path -> ground_truth_text
    cursor: dict {'pos': int} tracking how far into all_texts we've consumed
            for this pool; texts are picked sequentially (first N), not
            randomly.
    """
    total_planned = sum(n for _, n in plan)
    total_generated = 0

    with tqdm(total=total_planned, desc=desc, unit="img") as pbar:
        for folder, n in plan:
            texts_for_folder = next_texts(all_texts, cursor, n)
            style_name = folder.name
            pbar.set_postfix(folder=style_name)

            # Generate one image at a time so the progress bar advances
            # per-image instead of jumping once per folder batch.
            for i, text in enumerate(texts_for_folder):
                generate_images(
                    model=model,
                    vae=vae,
                    style_folder=str(folder),
                    texts=[text],
                    device=device,
                    epoch=tag,
                    step=None,
                    out_dir=out_dir,
                )

                # IMPORTANT: since we pass a single-text list each call,
                # generate_images almost certainly writes it as index 0
                # every time (epoch_{tag}_style_{style_name}_0.png).
                # We rename each output to a unique final name -- but if we
                # ever reused "_0.png" as a FINAL name too (i.e. when i==0),
                # that file would never actually get moved, and the NEXT
                # generate_images call would silently overwrite/delete it
                # (since it also writes to "_0.png"). Fix: number final
                # files starting at 1, so "_0.png" is ONLY ever the
                # temporary produced_path and never a final filename.
                produced_fname = f"epoch_{tag}_style_{style_name}_0.png"
                produced_path = os.path.join(out_dir, produced_fname)

                final_fname = f"epoch_{tag}_style_{style_name}_{i + 1}.png"
                final_path = os.path.join(out_dir, final_fname)

                if os.path.exists(produced_path):
                    os.replace(produced_path, final_path)
                else:
                    print(f"[WARN] expected output not found: {produced_path} "
                          f"(check generate_images' actual save-filename pattern)")

                # keyed to match the post-reorganization path used later:
                # eval_generated_2k_dcer_split/{tag}/{style_name}/{final_fname}
                rel_path = f"{tag}/{style_name}/{final_fname}"
                transcriptions[rel_path] = text

                pbar.update(1)
                total_generated += 1

    return total_generated


def main():
    model = Diffusion(font_path=FONT_PATH).to(device)
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device)
    vae.eval()

    load_checkpoint_for_inference(CKPT_PATH, model)
    model.eval()

    with open(TEXT_LIST_PATH, encoding="utf-8") as f:
        all_texts = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(all_texts)} candidate texts")

    print("\nPlanning TEST style folder generation...")
    test_plan = pick_folders_and_counts(TEST_STYLE_ROOT, TEST_PER_FOLDER, TEST_TARGET_TOTAL)
    print(f"Planned {sum(n for _, n in test_plan)} images across {len(test_plan)} test folders")

    print("\nPlanning TRAIN style folder generation...")
    train_plan = pick_folders_and_counts(TRAIN_STYLE_ROOT, TRAIN_PER_FOLDER, TRAIN_TARGET_TOTAL)
    print(f"Planned {sum(n for _, n in train_plan)} images across {len(train_plan)} train folders")

    transcriptions = {}

    # Separate cursors: TEST draws from the first TEST_TARGET_TOTAL lines of
    # the text file, TRAIN draws (independently) from the first
    # TRAIN_TARGET_TOTAL lines. Both start at line 0.
    SKIP_ROWS = 100000
    test_cursor = {"pos": SKIP_ROWS}
    train_cursor = {"pos": SKIP_ROWS}

    n_test = run_generation_for_pool(
        test_plan, all_texts, model, vae, tag="test", out_dir=OUT_DIR,
        desc="TEST styles", transcriptions=transcriptions, cursor=test_cursor
    )
    n_train = run_generation_for_pool(
        train_plan, all_texts, model, vae, tag="train", out_dir=OUT_DIR,
        desc="TRAIN styles", transcriptions=transcriptions, cursor=train_cursor
    )

    # Save transcriptions.json in OUT_DIR (used later once images are
    # reorganized into eval_generated_2k_dcer_split/{test,train}/{style}/*.png)
    transcriptions_path = os.path.join(OUT_DIR, "transcriptions.json")
    with open(transcriptions_path, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(transcriptions)} transcriptions to {transcriptions_path}")

    print(f"\nDone. Total generated: {n_test + n_train} images -> saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
