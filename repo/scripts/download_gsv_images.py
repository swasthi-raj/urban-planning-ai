"""
Download Google Street View images for a list of addresses.

SETUP:
1. pip install pandas requests openpyxl
2. Set your API key as an environment variable (DO NOT hardcode it):
     Mac/Linux:  export GSV_API_KEY="your_key_here"
     Colab:      os.environ["GSV_API_KEY"] = "your_key_here"  (in a separate cell, not committed)
3. Run: python download_gsv_images.py

INPUT: crack_addresses_6_neighborhoods.xlsx (must have 'lat', 'lng',
       'neighborhood_filter', 'service_request_id' columns)
OUTPUT: images saved into ./gsv_images/<neighborhood>/<id>.jpg
"""

import os
import time
import requests
import pandas as pd

# ---- CONFIG ----
API_KEY = os.environ.get("GSV_API_KEY")   # reads from environment, never hardcoded
INPUT_FILE = "crack_addresses_6_neighborhoods.xlsx"
OUTPUT_DIR = "gsv_images"
IMAGE_SIZE = "640x640"
FOV = 90
PITCH = 0
HEADINGS = [0, 90, 180, 270]

BASE_URL = "https://maps.googleapis.com/maps/api/streetview"


def download_image(lat, lng, heading, save_path):
    params = {
        "size": IMAGE_SIZE,
        "location": f"{lat},{lng}",
        "heading": heading,
        "pitch": PITCH,
        "fov": FOV,
        "key": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True
    else:
        print(f"  Failed: {save_path} (status {resp.status_code})")
        return False


def main():
    if not API_KEY:
        raise SystemExit(
            "No API key found. Set it first:\n"
            '  export GSV_API_KEY="your_key_here"   (Mac/Linux)\n'
            '  os.environ["GSV_API_KEY"] = "your_key_here"   (Colab, separate cell)'
        )

    df = pd.read_excel(INPUT_FILE)
    df = df.dropna(subset=["lat", "lng"])

    total = 0
    success = 0

    for _, row in df.iterrows():
        neighborhood = str(row["neighborhood_filter"]).replace(" ", "_").replace("(", "").replace(")", "")
        req_id = row["service_request_id"]
        lat, lng = row["lat"], row["lng"]

        folder = os.path.join(OUTPUT_DIR, neighborhood)
        os.makedirs(folder, exist_ok=True)

        for heading in HEADINGS:
            filename = f"{req_id}_h{heading}.jpg"
            save_path = os.path.join(folder, filename)
            total += 1
            if download_image(lat, lng, heading, save_path):
                success += 1
            time.sleep(0.1)

    print(f"\nDone. {success}/{total} images downloaded into ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
