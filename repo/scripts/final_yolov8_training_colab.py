# ============================================================
# FINAL YOLOv8-seg TRAINING — Urban Infrastructure Equity Study
# Run each cell in order in Google Colab (use a GPU runtime:
# Runtime > Change runtime type > T4 GPU)
# ============================================================

# ---- CELL 1: Install packages ----
!pip install ultralytics roboflow -q

# ---- CELL 2: Download your dataset from Roboflow ----
from roboflow import Roboflow

ROBOFLOW_API_KEY = "PASTE_YOUR_API_KEY_HERE"   # same key you used before

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("new-workspace-wsp49").project("urban_planning_ai-ugy33")
version = project.version(7)   # version number we just generated
dataset = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)

# ---- CELL 3: Train YOLOv8m-seg ----
from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")   # medium segmentation model, matches your prior Roboflow runs

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,          # let it run the full schedule; early stopping (patience) will kick in if it plateaus
    imgsz=640,
    patience=20,         # stop early if no improvement for 20 epochs
    batch=16,
    project="urban_infra_final",
    name="yolov8m_seg_run1",
)

# ---- CELL 4: Print overall results ----
print("\n=== OVERALL METRICS ===")
metrics = model.val()
print(f"mAP50: {metrics.seg.map50:.4f}")
print(f"mAP50-95: {metrics.seg.map:.4f}")
print(f"Precision: {metrics.seg.mp:.4f}")
print(f"Recall: {metrics.seg.mr:.4f}")

# ---- CELL 5: Print per-class results (this is what you need for H4) ----
print("\n=== PER-CLASS mAP50 ===")
class_names = model.names
for i, name in class_names.items():
    try:
        print(f"{name}: {metrics.seg.ap50[i]:.4f}")
    except Exception:
        print(f"{name}: (no instances in test set)")

# ---- CELL 6: Download the trained model + results ----
import shutil
shutil.make_archive("yolov8_final_results", "zip", "urban_infra_final/yolov8m_seg_run1")

from google.colab import files
files.download("yolov8_final_results.zip")
