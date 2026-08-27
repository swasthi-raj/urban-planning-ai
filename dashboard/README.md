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

## Model

YOLOv8-seg training notebook: https://colab.research.google.com/drive/1rIZhge7mEPGKe_ebf4KbcLQaIWR-7dry

## Dashboard — Streamlit shell

Streamlit shell for the project. Currently running on **placeholder data**
so the layout, filters, and charts can be reviewed and iterated on while
the v4 model finishes real neighborhood inference.

### Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

### What's in it

- **KPI row** — total detections, severe/light counts, top equity-score neighborhood
- **Map** — bubble map of the 7 neighborhoods, sized by equity score, colored by Underserved/Wealthier
- **Class breakdown chart** — detections by infrastructure class
- **Neighborhood summary table** — detections, severity split, equity score, 311 complaints
- **H5 comparison** — AI-detected counts vs. 311 reported counts per neighborhood

### Swapping in real data

Everything placeholder lives in the **DATA LAYER** section at the top of `app.py`,
in two functions:

| Function | Replace with |
|---|---|
| `load_placeholder_detections()` | Real YOLOv8-seg inference output across the 7 neighborhoods. Needs columns: `neighborhood`, `category`, `class_name`, `severity`, `confidence`, `lat`, `lon` |
| `load_placeholder_311()` | Real San Diego 311 Get It Done complaint counts. Needs: `neighborhood`, `reported_311_complaints` |

As long as the replacement functions return DataFrames with the same column
names, nothing else in the file needs to change — the UI, map, and charts all
read from these functions.

The `SEVERITY_MAP` dictionary controls the Severe/Light grouping
(Pothole + Alligator Crack = Severe, Longitudinal + Transverse Crack = Light).
Update this if the severity grouping logic changes.

The `compute_equity_score()` function has a placeholder weighting formula
(Severe = 2x, Light = 1x). **This should be finalized with Dr. Fernandez**
before it's used in any real reporting — flagged in the code as a TODO-style
comment.

### Known limitations to keep in mind

- Alligator Crack and Transverse Crack have lower model accuracy (mAP50 ~0.32
  each per the v4 training run), so real Severe/Light counts once plugged in
  will likely be undercounts, particularly for Severe, since Alligator Crack
  feeds into that bucket.
