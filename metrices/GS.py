"""
compute_gs.py
Computes Geometry Score for both splits. reals_train/fakes_train are capped
by truncating the FolderDataset's internal lists BEFORE unfolding, since
SubsetDataset doesn't support unfold() and GeometryBackbone loads all
patches into memory at once (no batching/streaming).

-------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------
1. Requirements (install once):
       pip install hwd

2. Update the path constants near the top of this file:
       REAL_TRAIN_DIR -> folder of real handwriting images for the TRAIN
                          style split
       fakes_train_ds -> folder of generated ("fake") images to compare
                          against REAL_TRAIN_DIR
   The commented-out REAL_TEST_DIR / fakes_test / reals_test lines show
   how to also score a TEST split the same way — uncomment and fill in
   paths if you want that comparison too.

3. (Optional) Adjust REAL_TRAIN_CAP — the number of samples each dataset
   is randomly capped to BEFORE unfold() runs. GeometryBackbone loads all
   patches into memory at once with no streaming, so this cap exists to
   avoid running out of memory on large datasets; raise it if you have
   the RAM, or lower it if you still hit OOM.

4. Run it:
       python compute_gs.py

   Output:
       - Progress messages while each split is capped and unfolded
       - Printed Geometry Score for the train split (and test split too,
         if you re-enable those lines)
-------------------------------------------------------------------------
"""

import random
from hwd.datasets import FolderDataset
from hwd.scores import GeometryScore

# REAL_TEST_DIR = "/home/kishan/diffusion/output_dataset_hindi_with_json_line_64*1024/test"
REAL_TRAIN_DIR = "/home/kishan/diffusion/output_dataset_hindi_with_json_line_64*1024/train"

REAL_TRAIN_CAP = 2000   # cap BEFORE unfold to avoid OOM
random.seed(42)


def cap_dataset(dataset, max_samples):
    """Randomly truncate a BaseDataset's imgs/authors/labels in place."""
    if len(dataset.imgs) <= max_samples:
        return dataset
    idx = random.sample(range(len(dataset.imgs)), max_samples)
    dataset.imgs = [dataset.imgs[i] for i in idx]
    dataset.authors = [dataset.authors[i] for i in idx]
    dataset.labels = [dataset.labels[i] for i in idx]
    return dataset


print("Starting Geometry Score computation for both splits...")

# fakes_test = FolderDataset("eval_generated_hwd_split/test").unfold(verbose=True)
# reals_test = FolderDataset(REAL_TEST_DIR).unfold(verbose=True)
# print("Test unfold done.")

fakes_train_ds = FolderDataset("/home/kishan/diffusion/Syn_Data/syn_data_grouped")
fakes_train_ds = cap_dataset(fakes_train_ds, REAL_TRAIN_CAP)
fakes_train = fakes_train_ds.unfold(verbose=True)

reals_train_ds = FolderDataset(REAL_TRAIN_DIR)
reals_train_ds = cap_dataset(reals_train_ds, REAL_TRAIN_CAP)
reals_train = reals_train_ds.unfold(verbose=True)
print("Train unfold done (capped).")

gs = GeometryScore(height=32)

# print("Computing GS for test split...")
# score_test = gs(fakes_test, reals_test, verbose=True)
# print(f"GS (test-style generated vs real TEST styles): {score_test}")

print("Computing GS for train split (capped)...")
score_train = gs(fakes_train, reals_train, verbose=True)
print(f"GS (train-style generated vs real TRAIN styles): {score_train}")
