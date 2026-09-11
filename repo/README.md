# AI-Powered Urban Infrastructure Monitoring — San Diego County

An AI-enabled geospatial framework for assessing street-level infrastructure conditions
across San Diego neighborhoods, combining computer vision detection with San Diego's
311 "Get It Done" civic reporting data to identify spatial equity gaps.

**Faculty Advisor:** Dr. Gabriela Fernandez
**Team:** Swasthika Rajendran, Rachel Hamaker, Sayali Sanjay Shelke
**Lab:** SDSU MOCL Lab (Metabolism of Cities Living Lab)

## Live Dashboard

Deployed via Streamlit Community Cloud: *(add your live link here after deployment)*

## Project Structure

```
├── streamlit_app.py          # Main interactive dashboard (Streamlit)
├── dashboard.html            # Standalone HTML version of the dashboard
├── requirements.txt          # Python dependencies
├── scripts/
│   ├── download_gsv_images.py       # Pulls Google Street View images from address lists
│   ├── export_labels_only.py        # Exports YOLO-format labels from Roboflow for local processing
│   └── final_yolov8_training_colab.py  # Colab training script for the final YOLOv8-seg model
└── data/
    ├── 311_shortlist_6_neighborhoods.xlsx      # Sampled 311 complaint addresses
    ├── crack_addresses_6_neighborhoods.xlsx    # Crack-specific 311 complaint addresses
    ├── pothole_addresses_6_neighborhoods.xlsx  # Pothole-specific 311 complaint addresses
    └── random_street_sample_7_neighborhoods.xlsx  # Randomly generated street coordinates
```

## Methodology Summary

1. **Image collection:** Google Street View Static API, sampled from (a) San Diego 311
   complaint addresses (general, pavement, and pothole-specific) and (b) randomly
   generated coordinates across 7 neighborhoods, for an unbiased comparison baseline.
2. **Annotation:** Manual polygon (instance segmentation) labeling in Roboflow across
   5 classes — Pothole, Light Crack, Severe Crack, Sidewalk, Bike Lane — supplemented
   with public datasets for underrepresented classes (training set only).
3. **Model:** YOLOv8-seg (primary model), benchmarked against RF-DETR-seg-medium.
   Final model: 61.9% mAP@50 overall.
4. **Equity analysis:** A weighted Equity Index (infrastructure presence vs. weighted
   damage) computed on a random, unbiased street sample per neighborhood, cross-referenced
   against median household income (SANDAG/Census) and 311 complaint volume.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploying

See [Streamlit Community Cloud](https://share.streamlit.io) — connect this repo,
set the main file to `streamlit_app.py`, and deploy.

## Data Sources

- Google Street View Static API
- San Diego 311 "Get It Done" Open Data Portal (data.sandiego.gov)
- SANDAG / U.S. Census median household income estimates (2022)
