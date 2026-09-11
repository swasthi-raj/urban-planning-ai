# ============================================================
# EXPORT LABELS ONLY (small file, no images) — run this in Colab
# ============================================================

import os

# ---- CELL 1: Check if dataset is already downloaded (from your training session) ----
import glob
existing = glob.glob("/content/Urban_Planning_AI-7*")
print("Found existing dataset folders:", existing)

# ---- CELL 2: If nothing found above, re-download it (only needed if your session restarted) ----
if not existing:
    from roboflow import Roboflow

    ROBOFLOW_API_KEY = "PASTE_YOUR_API_KEY_HERE"

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("new-workspace-wsp49").project("urban_planning_ai-ugy33")
    version = project.version(7)
    dataset = version.download("yolov8")
    dataset_path = dataset.location
    print("Downloaded fresh to:", dataset_path)
else:
    dataset_path = existing[0]
    print("Using existing dataset at:", dataset_path)

# ---- CELL 3: Zip just the labels folders (small, text-only files) ----
import shutil

zip_name = "labels_only"
shutil.make_archive(zip_name, "zip", dataset_path, base_dir=".")

# Actually we only want the label .txt files, not images — do it more precisely:
import zipfile

with zipfile.ZipFile("labels_only.zip", "w") as zf:
    for split in ["train", "valid", "test"]:
        label_dir = os.path.join(dataset_path, split, "labels")
        if os.path.exists(label_dir):
            for fname in os.listdir(label_dir):
                filepath = os.path.join(label_dir, fname)
                arcname = os.path.join(split, "labels", fname)
                zf.write(filepath, arcname)
            print(f"Added {len(os.listdir(label_dir))} files from {split}/labels")
        else:
            print(f"WARNING: {label_dir} not found")

print("\nDone! labels_only.zip created.")

# ---- CELL 4: Download it ----
from google.colab import files
files.download("labels_only.zip")
