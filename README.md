# Urban Infrastructure Equity Dashboard

## Abstract

Cities face persistent challenges in monitoring and maintaining critical
infrastructure, particularly in underserved communities where limited
resources and infrequent field inspections delay the identification of
infrastructure deficiencies. Advances in artificial intelligence (AI),
computer vision, and geospatial technologies provide new opportunities for
scalable, equitable infrastructure assessment. This study presents an
AI-driven framework for automated urban infrastructure assessment in San
Diego County, CA using Google Street View imagery and deep learning-based
object detection. Images were collected through systematic sampling with
the Google Street View Static API and targeted retrieval at locations
identified through San Diego's 311 Get It Done civic complaint platform.
Integrating citizen-reported complaints with AI-detected infrastructure
conditions enables cross-validation of reported issues and observed
conditions across neighborhoods.

Street-level images were manually annotated to identify infrastructure
assets and deficiencies, including potholes, light and severe road cracks,
sidewalks, bicycle lanes, and other built environment features.
Polygon-based instance segmentation annotations were used to train a
YOLOv8 model capable of accurately identifying and localizing
infrastructure conditions across diverse urban contexts.

The proposed framework reduces the time and cost of traditional field
surveys while improving the spatial coverage and consistency of
infrastructure assessments. By generating scalable and repeatable
measurements, the approach supports the identification of infrastructure
disparities, enabling data-driven decision-making and equity-focused
planning. Integrating computer vision with geospatial analytics
facilitates infrastructure inventories and priority maps for targeted
investments in vulnerable communities. This research demonstrates the
potential of AI-enabled street-level imagery as a cost-effective tool for
infrastructure monitoring. Future research will integrate video collected
from municipal service vehicles to support continuous, near-real-time
urban infrastructure assessment.

## Model & training results

YOLOv8-seg training notebook: https://colab.research.google.com/drive/1rIZhge7mEPGKe_ebf4KbcLQaIWR-7dry

Current production weights: `yolov8m-seg`, 60 epochs, 6 classes.

| Class | mAP50 |
|---|---|
| Bike_Lane | 0.921 |
| Side_Walk | 0.847 |
| Longitudinal Crack | 0.550 |
| Pothole | 0.501 |
| Alligator Crack | 0.328 |
| Transverse Crack | 0.318 |
| **All classes** | **0.577** |

Alligator Crack and Transverse Crack are the weakest classes — the
normalized confusion matrix shows both are frequently confused with
background (0.47 and 0.22 of true instances misclassified as background,
respectively), so real Severe/Light counts are likely undercounts,
particularly Severe (Alligator Crack feeds into that bucket). This should
be kept in mind when interpreting equity scores until the model is
retrained with more examples of those two classes.

## Data pipeline

1. **`process_311.py`** — downloads and filters San Diego's Get It Done
   311 dataset down to the 7 study neighborhoods (no API key needed).
   El Cajon is excluded — it's a separate incorporated city, not covered
   by San Diego's 311 system.
2. **`download_gsv.py`** — pulls new Google Street View imagery across the
   7 neighborhoods via the Street View Static API (needs a `GOOGLE_MAPS_API_KEY`
   in a local `.env`).
3. **`run_inference.py`** — runs the trained model over geotagged imagery
   and writes `data/detections.csv` in the schema the dashboard reads.

The current `data/detections.csv` was produced from the geotagged subset
of the annotated training/validation image set (images whose filenames
carry real lat/lon from prior Street View collection), snapped to the
nearest of the 7 study neighborhoods within a 3km radius. Anything farther
than that was dropped rather than force-assigned.

## Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

### What's in it

- **KPI row** — total detections, severe/light counts, top equity-score neighborhood
- **Map** — bubble map of the 7 neighborhoods, sized by equity score, colored by Underserved/Wealthier
- **Class breakdown chart** — detections by infrastructure class
- **Neighborhood summary table** — detections, severity split, equity score, 311 complaints
- **H5 comparison** — AI-detected counts vs. 311 reported counts per neighborhood (El Cajon excluded — no 311 coverage)

`app.py` automatically uses real data from `data/detections.csv` and
`data/complaints_311.csv` when those files exist, and falls back to
seeded placeholder data otherwise — no code changes needed either way.

The `compute_equity_score()` function has a placeholder weighting formula
(Severe = 2x, Light = 1x). **This should be finalized with Dr. Fernandez**
before it's used in any real reporting.
