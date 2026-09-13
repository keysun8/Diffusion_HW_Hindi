"""
compute_hwd_fixed.py
Computes HWD separately for test-style-generated vs real-test,
and train-style-generated vs real-train — matching authors correctly.

Progress tracking:
HWDScore doesn't expose its own progress bar. It extracts features by
running each dataset through a DataLoader internally (batch loading +
forward pass through the backbone), so instead of guessing at internals,
we monkey-patch torch's DataLoader.__iter__ globally for this process.
Every DataLoader created anywhere (including inside HWDScore) gets
wrapped in tqdm automatically. Because each loop iteration's elapsed
time includes both fetching the next batch AND whatever the previous
iteration's body did (the forward pass), this gives a real end-to-end
progress bar — not just a data-loading one.

-------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------
1. Requirements (install once):
       pip install torch tqdm hwd

   (`hwd` is the package providing FolderDataset / HWDScore — install
   whatever distribution you're using for it if the plain `pip install hwd`
   name doesn't match.)

2. Update the path constants near the top of this file:
       REAL_TRAIN_DIR -> folder of real handwriting images for the TRAIN
                          style split
       fakes_train    -> folder of generated ("fake") images to compare
                          against REAL_TRAIN_DIR
   The commented-out REAL_TEST_DIR / fakes_test / reals_test lines show
   how to also score a TEST split the same way — uncomment and fill in
   paths if you want that comparison too.

3. (Optional) Adjust `HWDScore(height=32)` if you want the images resized
   to a different height before feature extraction.

4. Run it:
       python compute_hwd_fixed.py

   Output:
       - A tqdm progress bar while features are extracted (via the
         monkey-patched DataLoader, so it reflects real end-to-end
         progress, not just data loading)
       - Printed dataset sizes (fakes_train / reals_train, and the test
         split too if you re-enable those lines)
       - Printed HWD score(s) to the console
-------------------------------------------------------------------------
"""

import torch
from tqdm import tqdm

from hwd.datasets import FolderDataset
from hwd.scores import HWDScore

# --- Monkey-patch: wrap every DataLoader iteration in a tqdm bar ---
_original_iter = torch.utils.data.DataLoader.__iter__


def _tqdm_wrapped_iter(self):
    try:
        total = len(self)
    except TypeError:
        total = None
    return iter(tqdm(_original_iter(self), total=total, desc="HWD processing", leave=True))


torch.utils.data.DataLoader.__iter__ = _tqdm_wrapped_iter
# --------------------------------------------------------------------

# REAL_TEST_DIR = "/home/kishan/diffusion/Syn_Data/hw_restructured/train"
REAL_TRAIN_DIR = "/home/kishan/diffusion/output_dataset_hindi_with_json_line_64*1024/train"

# fakes_test = FolderDataset("/home/kishan/diffusion/Syn_Data/hw_syn_1")
fakes_train = FolderDataset("/home/kishan/diffusion/Syn_Data/syn_data_grouped")
# reals_test = FolderDataset(REAL_TEST_DIR)
reals_train = FolderDataset(REAL_TRAIN_DIR)

print(f"fakes_train: {len(fakes_train)}, reals_train: {len(reals_train)}")

hwd = HWDScore(height=32)

# score_test = hwd(fakes_test, reals_test)
# print(f"HWD (test-style generated vs real TEST styles): {score_test}")

score_train = hwd(fakes_train, reals_train)
print(f"HWD (train-style generated vs real TRAIN styles): {score_train}")
