"""
compute_fid.py
Computes FID (standard Frechet Inception Distance) for both splits,
reusing the same FolderDataset structure built for HWD.

-------------------------------------------------------------------------
HOW TO RUN
-------------------------------------------------------------------------
1. Requirements (install once):
       pip install torch hwd

   (`hwd` is the package providing FolderDataset / FIDScore — install
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

3. (Optional) Adjust `FIDScore(height=32)` if you want the images resized
   to a different height before feature extraction (32 matches HWD's
   convention).

4. Run it:
       python compute_fid.py

   Output:
       - Printed dataset sizes (fakes_train / reals_train, and the test
         split too if you re-enable those lines)
       - Printed FID score(s) to the console
-------------------------------------------------------------------------
"""

from hwd.datasets import FolderDataset
from hwd.scores import FIDScore

# REAL_TEST_DIR = "/home/kishan/diffusion/Syn_Data/hw_restructured/train"
REAL_TRAIN_DIR = "/home/kishan/diffusion/output_dataset_hindi_with_json_line_64*1024/train"

print("start")

# fakes_test = FolderDataset("/home/kishan/diffusion/Syn_Data/hw_syn_1")
fakes_train = FolderDataset("/home/kishan/diffusion/Syn_Data/syn_data_grouped")
# reals_test = FolderDataset(REAL_TEST_DIR)
reals_train = FolderDataset(REAL_TRAIN_DIR)

# print(f"fakes_test: {len(fakes_test)}, reals_test: {len(reals_test)}")
print(f"fakes_train: {len(fakes_train)}, reals_train: {len(reals_train)}")

fid = FIDScore(height=32)  # matches HWD's height convention; adjust if you want higher-res crops

# score_test = fid(fakes_test, reals_test)
# print(f"FID (test-style generated vs real TEST styles): {score_test}")

score_train = fid(fakes_train, reals_train)
print(f"FID (train-style generated vs real TRAIN styles): {score_train}")
