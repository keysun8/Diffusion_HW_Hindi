"""
word_segmentation.py , using easy ocr 

Splits synthetic handwriting-line images (each containing a known number
of words, per their ground-truth transcription) into individual per-word
crops, using gaps in the vertical ink projection profile to find word
boundaries. Also writes out a word-level ground-truth JSON mapping each
cropped filename to its corresponding word.

-------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------
1. Requirements (install once):
       pip install opencv-python numpy tqdm easyocr

2. Update the path constants near the top of this file:
       image_path            -> folder (searched recursively) containing
                                 the full line images to segment
       word_image_output_path -> folder where per-word crops will be saved
                                 (created automatically if missing)
       gt_input_path         -> transcriptions.json mapping each line
                                 image's filename -> its full ground-truth
                                 sentence/line text
       gt_output_path        -> where the new word-level ground-truth JSON
                                 will be written

3. (Optional) Tune the segmentation sensitivity:
       MIN_GAP_PX     -> minimum gap width (in pixels) to count as a
                          candidate word boundary
       GAP_NOISE_FRAC -> a column is treated as "empty" (part of a gap) if
                          its ink sum is <= this fraction of the max
                          column-ink value in the image

   Note: this script currently assumes every line has exactly 3 words
   (see the `len(words_in_sentence) != 3` check) — adjust that if your
   data uses a different word count.

4. Run it:
       python word_segmentation.py

   Output:
       - Per-word PNG crops saved into word_image_output_path, named
         {original_filename}_{word_index}.png
       - gt_output_path JSON mapping each crop filename to its word
       - Console summary: images saved, successfully segmented,
         GT word-count mismatches, missing GT, and unrecoverable images
         (not enough separable gaps found).
-------------------------------------------------------------------------
"""

import os
import json
import cv2
import numpy as np
from tqdm import tqdm

image_path = "/home/kishan/diffusion/Syn_Data/syn_hw_merged"
word_image_output_path = "/home/kishan/diffusion/Syn_Data/syn_hw_merged_words"
gt_input_path = "/home/kishan/diffusion/Syn_Data/syn_hw_merged/transcriptions.json"
gt_output_path = "/home/kishan/diffusion/Syn_Data/syn_hw_merged_words.json"

valid_extensions = ('.png', '.jpg', '.jpeg')
MIN_GAP_PX = 3          # minimum gap width (px) to count as a candidate word boundary
GAP_NOISE_FRAC = 0.02   # columns with ink <= this fraction of max column-ink are "empty"

os.makedirs(word_image_output_path, exist_ok=True)


def segment_by_projection(image_gray, n_words, min_gap_px=MIN_GAP_PX):
    """
    Split a single-line word image into exactly n_words segments using
    the widest whitespace gaps in the vertical ink projection profile.
    Returns None if fewer than n_words-1 usable gaps exist.
    """
    if image_gray.size == 0 or n_words <= 0:
        return None

    _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_sum = binary.sum(axis=0)

    if col_sum.max() == 0:
        return None

    if n_words == 1:
        return [(0, image_gray.shape[1])]

    is_gap = col_sum <= (col_sum.max() * GAP_NOISE_FRAC)

    gaps = []
    start = None
    for x, g in enumerate(is_gap):
        if g and start is None:
            start = x
        elif not g and start is not None:
            if x - start >= min_gap_px:
                gaps.append((start, x, x - start))  # (gap_start, gap_end, width)
            start = None
    if start is not None and len(col_sum) - start >= min_gap_px:
        gaps.append((start, len(col_sum), len(col_sum) - start))

    if len(gaps) < n_words - 1:
        return None  # not enough separable gaps — can't recover this one

    # take the widest (n_words - 1) gaps, then restore left-to-right order
    gaps.sort(key=lambda g: -g[2])
    chosen = sorted(gaps[:n_words - 1], key=lambda g: g[0])

    split_points = [0] + [(g[0] + g[1]) // 2 for g in chosen] + [image_gray.shape[1]]

    if len(split_points) - 1 != n_words:
        return None

    # guard against degenerate zero-width segments from overlapping gap midpoints
    for i in range(len(split_points) - 1):
        if split_points[i + 1] <= split_points[i]:
            return None

    return [(split_points[i], split_points[i + 1]) for i in range(n_words)]


# ---------------- LOAD GROUND TRUTH ----------------
with open(gt_input_path, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

gt_lookup = {}
for k, v in gt_data.items():
    basename = os.path.basename(k)
    if basename in gt_lookup:
        print(f"Warning: duplicate basename '{basename}' in GT — keeping first occurrence.")
        continue
    gt_lookup[basename] = v

# ---------------- MAIN LOOP ----------------
word_gt = {}
no_final_words = 0
success_count = 0
mismatch_count = 0  # GT didn't actually have 3 words (sanity guard)
no_gt_count = 0
unrecoverable_count = 0

for root, _, files in os.walk(image_path):
    for image_filename in tqdm(files, desc=f"processing {root}..."):
        if not image_filename.lower().endswith(valid_extensions):
            continue

        image_path1 = os.path.join(root, image_filename)
        lookup_key = os.path.basename(image_filename)

        if lookup_key not in gt_lookup:
            no_gt_count += 1
            continue

        sentence = gt_lookup[lookup_key]
        words_in_sentence = sentence.replace("।", "").strip().split()

        if len(words_in_sentence) != 3:
            # sanity guard — shouldn't normally happen, but don't silently mis-split
            mismatch_count += 1
            continue

        image_cv = cv2.imread(image_path1)
        if image_cv is None:
            print(f"Could not read image: {image_path1}, skipping.")
            continue

        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        height, width = image_gray.shape

        segments = segment_by_projection(image_gray, len(words_in_sentence))
        if segments is None:
            print(f"Could not find 2 separable gaps for {lookup_key}, skipping.")
            unrecoverable_count += 1
            continue

        success_count += 1
        image_filename1 = os.path.splitext(image_filename)[0]

        for idx, (sx, ex) in enumerate(segments):
            extra_x = 5
            extra_y_top = 12     # more headroom for matras above shirorekha
            extra_y_bottom = 8   # some room for descenders below baseline

            x_start = max(0, sx - extra_x)
            y_start = max(0, 0 - extra_y_top)
            x_end = min(width, ex + extra_x)
            y_end = min(height, height + extra_y_bottom)

            crop_image = image_cv[y_start:y_end, x_start:x_end]
            if crop_image.size == 0:
                print(f"Empty crop for segment: {(sx, ex)} in {lookup_key}")
                continue

            crop_name = f"{image_filename1}_{idx + 1}.png"
            crop_path = os.path.join(word_image_output_path, crop_name)

            success = cv2.imwrite(crop_path, crop_image)
            if not success:
                print(f"Failed to write: {crop_path}")
                continue

            no_final_words += 1
            word_gt[crop_name] = words_in_sentence[idx]

with open(gt_output_path, 'w', encoding='utf-8') as f:
    json.dump(word_gt, f, ensure_ascii=False, indent=2)

print(f'no of final images saved: {no_final_words}')
print(f'images successfully segmented: {success_count}')
print(f'images with GT word count != 3 (skipped): {mismatch_count}')
print(f'images with no matching ground truth: {no_gt_count}')
print(f'images unrecoverable (not enough gaps found): {unrecoverable_count}')
print(f'word-level ground truth saved to {gt_output_path}')
print("Done")
