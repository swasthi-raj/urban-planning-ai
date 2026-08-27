"""
Download and filter San Diego "Get It Done" 311 data down to the 7 study
neighborhoods, producing the counts file the dashboard reads.

Source: https://data.sandiego.gov/datasets/get-it-done-311/
No API key required — these are public CSV downloads.

NOTE — El Cajon is skipped here on purpose (see config.py): it's a
separate incorporated city, not a City of San Diego community planning
area, so it has no comm_plan_name and never appears in this dataset.

Usage:
    python process_311.py

Outputs (into ./data/):
    311_raw/get_it_done_requests_open_datasd.csv   — raw download, cached
    311_raw/get_it_done_requests_closed_2026_datasd.csv
    complaints_311.csv                              — aggregated counts by
                                                        neighborhood, ready
                                                        for the dashboard
"""

import os
import sys

import pandas as pd
import requests

from config import (
    NEIGHBORHOODS,
    INFRASTRUCTURE_311_KEYWORDS,
    RAW_311_DIR,
    COMPLAINTS_311_CSV,
)

SOURCE_FILES = {
    "open": "https://seshat.datasd.org/get_it_done_reports/get_it_done_requests_open_datasd.csv",
    "closed_2026": "https://seshat.datasd.org/get_it_done_reports/get_it_done_requests_closed_2026_datasd.csv",
}


def download_source_files() -> list[str]:
    """Download each CSV in SOURCE_FILES to RAW_311_DIR if not already cached."""
    os.makedirs(RAW_311_DIR, exist_ok=True)
    local_paths = []
    for label, url in SOURCE_FILES.items():
        local_path = os.path.join(RAW_311_DIR, os.path.basename(url))
        if os.path.exists(local_path):
            print(f"[skip] {label}: already downloaded -> {local_path}")
            local_paths.append(local_path)
            continue
        print(f"[download] {label}: {url}")
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            print(f"  saved -> {local_path} ({os.path.getsize(local_path) / 1e6:.1f} MB)")
            local_paths.append(local_path)
        except requests.RequestException as e:
            print(f"  FAILED to download {label}: {e}", file=sys.stderr)
            print(
                "  If this fails from a sandboxed/proxied shell, download the file "
                "manually from https://data.sandiego.gov/datasets/get-it-done-311/ "
                f"and place it at {local_path}",
                file=sys.stderr,
            )
    return local_paths


def load_and_filter(paths: list[str]) -> pd.DataFrame:
    """Load raw 311 CSVs, filter to study neighborhoods + infrastructure categories."""
    comm_plan_lookup = {
        v["comm_plan_name"].upper(): name
        for name, v in NEIGHBORHOODS.items()
        if v["comm_plan_name"]
    }
    skipped = [name for name, v in NEIGHBORHOODS.items() if not v["comm_plan_name"]]
    if skipped:
        print(f"[info] skipping (no San Diego comm_plan_name): {', '.join(skipped)}")

    keyword_pattern = "|".join(INFRASTRUCTURE_311_KEYWORDS)

    frames = []
    for path in paths:
        print(f"[load] {path}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)

    if not frames:
        raise RuntimeError("No 311 source files loaded — nothing to filter.")

    all_requests = pd.concat(frames, ignore_index=True)
    all_requests["comm_plan_name"] = all_requests["comm_plan_name"].astype(str).str.upper().str.strip()

    neighborhood_matches = all_requests[all_requests["comm_plan_name"].isin(comm_plan_lookup.keys())].copy()
    print(f"[filter] {len(neighborhood_matches):,} rows in study neighborhoods "
          f"(of {len(all_requests):,} total)")

    text = (
        neighborhood_matches["service_name"].fillna("")
        + " "
        + neighborhood_matches["service_name_detail"].fillna("")
    ).str.lower()
    infra_matches = neighborhood_matches[text.str.contains(keyword_pattern, regex=True, na=False)].copy()
    print(f"[filter] {len(infra_matches):,} rows match infrastructure keywords "
          f"({INFRASTRUCTURE_311_KEYWORDS})")

    infra_matches["neighborhood"] = infra_matches["comm_plan_name"].map(comm_plan_lookup)
    return infra_matches


def aggregate_and_save(filtered: pd.DataFrame) -> pd.DataFrame:
    counts = (
        filtered.groupby("neighborhood")
        .size()
        .reset_index(name="reported_311_complaints")
    )
    # Ensure every study neighborhood has a row, even El Cajon (0, flagged as N/A-source).
    all_names = pd.DataFrame({"neighborhood": list(NEIGHBORHOODS.keys())})
    counts = all_names.merge(counts, on="neighborhood", how="left")
    counts["reported_311_complaints"] = counts["reported_311_complaints"].fillna(0).astype(int)
    counts["source"] = counts["neighborhood"].map(
        lambda n: "san_diego_311" if NEIGHBORHOODS[n]["comm_plan_name"] else "no_311_coverage"
    )

    os.makedirs(os.path.dirname(COMPLAINTS_311_CSV), exist_ok=True)
    counts.to_csv(COMPLAINTS_311_CSV, index=False)
    print(f"[save] {COMPLAINTS_311_CSV}")
    print(counts.to_string(index=False))
    return counts


if __name__ == "__main__":
    paths = download_source_files()
    if not paths:
        print("No source files available — nothing to process.", file=sys.stderr)
        sys.exit(1)
    filtered = load_and_filter(paths)
    aggregate_and_save(filtered)
