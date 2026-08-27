"""
Systematically download Google Street View Static API images across the 7
study neighborhoods, plus targeted pulls at San Diego 311 complaint
locations (run process_311.py first to get those).

Requires a Google Cloud API key with the Street View Static API enabled,
set as GOOGLE_MAPS_API_KEY (in a local .env file — see .env.example).

Usage:
    python download_gsv.py                  # grid sampling only
    python download_gsv.py --with-311       # grid sampling + 311-targeted

Images are saved to data/gsv_images/<neighborhood>/ with lat/lon encoded
in the filename, matching the pattern run_inference.py expects:
    streetview_<lat>_<lon>_heading<H>.jpg
"""

import argparse
import math
import os
import random
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

from config import (
    NEIGHBORHOODS,
    GOOGLE_MAPS_API_KEY,
    GSV_IMAGE_SIZE,
    GSV_SAMPLES_PER_NEIGHBORHOOD,
    GSV_NEIGHBORHOOD_RADIUS_METERS,
    GSV_TARGETED_311_LIMIT,
    IMAGES_DIR,
    COMPLAINTS_311_CSV,
)

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
HEADINGS = [0, 90, 180, 270]  # four compass directions per point, matching prior collection


def fname(lat, lon, heading):
    lat_s = f"{lat:.6f}".replace(".", "-")
    lon_s = f"{lon:.6f}".replace(".", "-")
    return f"streetview_{lat_s}_{lon_s}_heading{heading}.jpg"


def offset_point(lat, lon, radius_m):
    """Random point within radius_m meters of (lat, lon)."""
    r = radius_m * math.sqrt(random.random())
    theta = random.uniform(0, 2 * math.pi)
    dlat = (r * math.cos(theta)) / 111_320
    dlon = (r * math.sin(theta)) / (111_320 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def has_coverage(lat, lon) -> bool:
    resp = requests.get(METADATA_URL, params={
        "location": f"{lat},{lon}", "key": GOOGLE_MAPS_API_KEY,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json().get("status") == "OK"


def download_point(lat, lon, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if not has_coverage(lat, lon):
        return 0
    saved = 0
    for heading in HEADINGS:
        out_path = os.path.join(out_dir, fname(lat, lon, heading))
        if os.path.exists(out_path):
            saved += 1
            continue
        resp = requests.get(IMAGE_URL, params={
            "size": GSV_IMAGE_SIZE, "location": f"{lat},{lon}",
            "heading": heading, "key": GOOGLE_MAPS_API_KEY,
        }, timeout=30)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
            with open(out_path, "wb") as f:
                f.write(resp.content)
            saved += 1
    return saved


def grid_sample_neighborhood(name, info):
    out_dir = os.path.join(IMAGES_DIR, name.replace(" ", "_"))
    total_saved = 0
    print(f"[{name}] sampling {GSV_SAMPLES_PER_NEIGHBORHOOD} points within "
          f"{GSV_NEIGHBORHOOD_RADIUS_METERS}m of ({info['lat']}, {info['lon']})")
    for i in range(GSV_SAMPLES_PER_NEIGHBORHOOD):
        lat, lon = offset_point(info["lat"], info["lon"], GSV_NEIGHBORHOOD_RADIUS_METERS)
        saved = download_point(lat, lon, out_dir)
        total_saved += saved
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{GSV_SAMPLES_PER_NEIGHBORHOOD} points, {total_saved} images so far")
    print(f"[{name}] done — {total_saved} images")
    return total_saved


def targeted_311_sample():
    if not os.path.exists(COMPLAINTS_311_CSV):
        print("No data/complaints_311.csv found — run process_311.py first for --with-311.",
              file=sys.stderr)
        return
    import pandas as pd
    # complaints_311.csv currently holds counts, not individual lat/lon — this targeted
    # pass expects the per-row filtered file process_311.py can optionally also save
    # (extend process_311.py to write data/311_raw/infra_matches.csv with lat/lng columns
    # if you want per-complaint-location targeting instead of neighborhood-level counts).
    print("[info] Targeted 311-location retrieval needs per-complaint lat/lng — "
          "extend process_311.py to save the filtered row-level 311 data first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-311", action="store_true",
                         help="also pull images at 311 complaint locations (needs process_311.py run first)")
    args = parser.parse_args()

    if not GOOGLE_MAPS_API_KEY:
        print("GOOGLE_MAPS_API_KEY is not set. Put it in a local .env file "
              "(GOOGLE_MAPS_API_KEY=your_key_here) — see .env.example.", file=sys.stderr)
        sys.exit(1)

    grand_total = 0
    for name, info in NEIGHBORHOODS.items():
        grand_total += grid_sample_neighborhood(name, info)

    if args.with_311:
        targeted_311_sample()

    print(f"\nTotal images downloaded: {grand_total}")
    print(f"Saved under: {IMAGES_DIR}")
