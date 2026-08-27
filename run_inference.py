"""
Run the trained YOLOv8-seg model over street-view images and produce
data/detections.csv in the schema the dashboard reads:
neighborhood, category, class_name, severity, confidence, lat, lon.

Each image is assigned to a neighborhood one of two ways:

1. FOLDER NAME (preferred, no geotag needed) — if an image's immediate
   parent folder matches a neighborhood name from config.py (spaces or
   underscores, case-insensitive — "La Jolla", "La_Jolla", "la jolla" all
   match), it's assigned straight to that neighborhood at that
   neighborhood's center lat/lon. This is what makes adding a new
   neighborhood easy: drop images into data/gsv_images/<Name>/ (see
   download_gsv.py, which already saves into per-neighborhood folders)
   and run this script — no filename geotagging required.
2. FILENAME GEOTAG (fallback) — if the folder name doesn't match, lat/lon
   is parsed from the filename (patterns used by the original GSV
   collection and download_gsv.py's output):
     img_0152_N_32-7111_-117-0581_jpg.rf.<hash>.jpg
     streetview_32-5837496_-117-0924478_jpg.rf.<hash>.jpg
     streetview_<lat>_<lon>_heading<H>.jpg
   and snapped to the nearest study neighborhood within MAX_DISTANCE_KM.

Usage:
    python run_inference.py --weights "best (3).pt" --images-dir path/to/images
    python run_inference.py --weights "best (3).pt"   # scans IMAGES_DIR recursively

Images that match neither path are skipped (written to
data/skipped_images.csv with a reason) rather than force-assigned.

Writing is MERGE-SAFE: existing rows in data/detections.csv for
neighborhoods NOT touched by this run are preserved. Only the
neighborhoods that had images processed this run have their rows
replaced — so running inference on one new neighborhood's images never
wipes out the other neighborhoods' existing detections.
"""

import argparse
import math
import os
import re
import sys

import pandas as pd

from config import (
    NEIGHBORHOODS,
    SEVERITY_MAP,
    IMAGES_DIR,
    DETECTIONS_CSV,
    DATA_DIR,
)

# The trained model has one misspelled class name from the original Roboflow
# export ("Logitudinal Crack") — map it to the correct display name without
# touching the model itself.
MODEL_CLASS_DISPLAY_NAME = {"Logitudinal Crack": "Longitudinal Crack"}

MAX_DISTANCE_KM = 3.0
SKIPPED_CSV = os.path.join(DATA_DIR, "skipped_images.csv")

PATTERNS = [
    re.compile(r"img_\d+_[NSEW]_(?P<lat>\d+-\d+)_(?P<lon>-?\d+-\d+)_jpg"),
    re.compile(r"streetview_(?P<lat>\d+-\d+)_(?P<lon>-?\d+-\d+)"),
]


def folder_name_to_neighborhood(path: str):
    """
    Check whether the image's immediate parent folder name matches a
    neighborhood key from config.py (spaces/underscores and case
    ignored). Returns (name, category) or None if no match.
    """
    folder = os.path.basename(os.path.dirname(path))
    normalized = folder.replace("_", " ").strip().lower()
    for name, info in NEIGHBORHOODS.items():
        if name.strip().lower() == normalized:
            return name, info["category"]
    return None


def parse_latlon(filename: str):
    for pat in PATTERNS:
        m = pat.search(filename)
        if m:
            lat = float(m.group("lat").replace("-", ".", 1))
            lon_raw = m.group("lon")
            neg = lon_raw.startswith("-")
            if neg:
                lon_raw = lon_raw[1:]
            lon = float(lon_raw.replace("-", ".", 1))
            if neg:
                lon = -lon
            return lat, lon
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_neighborhood(lat, lon):
    best_name, best_dist, best_cat = None, float("inf"), None
    for name, info in NEIGHBORHOODS.items():
        d = haversine_km(lat, lon, info["lat"], info["lon"])
        if d < best_dist:
            best_name, best_dist, best_cat = name, d, info["category"]
    return best_name, best_dist, best_cat


def find_images(images_dir):
    found = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                found.append(os.path.join(root, f))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to trained .pt weights")
    parser.add_argument("--images-dir", default=IMAGES_DIR, help="folder to scan recursively for images")
    parser.add_argument("--max-distance-km", type=float, default=MAX_DISTANCE_KM)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO  # imported here so --help works without ultralytics installed

    if not os.path.exists(args.weights):
        print(f"Weights file not found: {args.weights}", file=sys.stderr)
        sys.exit(1)

    files = find_images(args.images_dir)
    if not files:
        print(f"No images found under {args.images_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(files)} images found under {args.images_dir}")

    kept_meta, skipped = [], []
    for path in files:
        fname = os.path.basename(path)

        # 1. Folder-name match — preferred, no geotag needed.
        folder_match = folder_name_to_neighborhood(path)
        if folder_match:
            name, cat = folder_match
            info = NEIGHBORHOODS[name]
            kept_meta.append({"file": fname, "path": path, "lat": info["lat"], "lon": info["lon"],
                               "neighborhood": name, "category": cat})
            continue

        # 2. Fallback — parse lat/lon from filename, snap to nearest neighborhood.
        parsed = parse_latlon(fname)
        if not parsed:
            skipped.append({"file": fname, "reason": "no_folder_match_and_no_latlon_in_filename"})
            continue
        lat, lon = parsed
        name, dist_km, cat = nearest_neighborhood(lat, lon)
        if dist_km > args.max_distance_km:
            skipped.append({"file": fname, "reason": f"too_far_{dist_km:.1f}km_from_{name}"})
            continue
        kept_meta.append({"file": fname, "path": path, "lat": lat, "lon": lon,
                           "neighborhood": name, "category": cat})

    print(f"{len(kept_meta)} images kept, {len(skipped)} skipped (see {SKIPPED_CSV})")
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(skipped).to_csv(SKIPPED_CSV, index=False)

    if not kept_meta:
        print("Nothing to run inference on.")
        return

    model = YOLO(args.weights)
    model_names = model.names
    print("Model classes:", model_names)

    rows = []
    for i in range(0, len(kept_meta), args.batch_size):
        batch = kept_meta[i:i + args.batch_size]
        results = model.predict([b["path"] for b in batch], conf=args.conf, verbose=False)
        for meta, result in zip(batch, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                raw_name = model_names[int(cls_id)]
                class_name = MODEL_CLASS_DISPLAY_NAME.get(raw_name, raw_name)
                rows.append({
                    "neighborhood": meta["neighborhood"],
                    "category": meta["category"],
                    "class_name": class_name,
                    "severity": SEVERITY_MAP.get(class_name, "N/A"),
                    "confidence": round(float(conf), 3),
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "source_image": meta["file"],
                })
        print(f"  processed {min(i + args.batch_size, len(kept_meta))}/{len(kept_meta)}")

    new_df = pd.DataFrame(rows)
    touched_neighborhoods = {m["neighborhood"] for m in kept_meta}

    # Merge-safe write: keep existing rows for neighborhoods this run did
    # NOT process (e.g. adding one new neighborhood's images shouldn't wipe
    # out the other neighborhoods' already-computed detections).
    os.makedirs(os.path.dirname(DETECTIONS_CSV), exist_ok=True)
    if os.path.exists(DETECTIONS_CSV):
        existing_df = pd.read_csv(DETECTIONS_CSV)
        preserved_df = existing_df[~existing_df["neighborhood"].isin(touched_neighborhoods)]
        df = pd.concat([preserved_df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(DETECTIONS_CSV, index=False)
    print(f"This run: {len(new_df)} detections for {sorted(touched_neighborhoods)}")
    print(f"Wrote {len(df)} total detections across {df['neighborhood'].nunique()} neighborhoods -> {DETECTIONS_CSV}")
    print(df.groupby("neighborhood").size())

    missing = [n for n in NEIGHBORHOODS if n not in set(df["neighborhood"])]
    if missing:
        print(f"No detections for: {', '.join(missing)} — dashboard will flag these as 'no data yet'.")


if __name__ == "__main__":
    main()
