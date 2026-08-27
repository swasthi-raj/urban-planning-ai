# Urban Infrastructure Equity Dashboard — Shell

Streamlit shell for the AI-Powered Urban Infrastructure Monitoring project.
Currently running on **placeholder data** so the layout, filters, and charts
can be reviewed and iterated on while the v4 model finishes real neighborhood
inference.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## What's in it

- **KPI row** — total detections, severe/light counts, top equity-score neighborhood
- **Map** — bubble map of the 7 neighborhoods, sized by equity score, colored by Underserved/Wealthier
- **Class breakdown chart** — detections by infrastructure class
- **Neighborhood summary table** — detections, severity split, equity score, median income, 311 complaints
- **H2 scatter** — income vs. equity score, for the equity/environmental justice hypothesis
- **H5 comparison** — AI-detected counts vs. 311 reported counts per neighborhood

## Swapping in real data

Everything placeholder lives in the **DATA LAYER** section at the top of `app.py`,
in three functions:

| Function | Replace with |
|---|---|
| `load_placeholder_detections()` | Real YOLOv8-seg inference output across the 7 neighborhoods. Needs columns: `neighborhood`, `category`, `class_name`, `severity`, `confidence`, `lat`, `lon` |
| `load_placeholder_socioeconomic()` | Real Census ACS pull. Needs: `neighborhood`, `category`, `median_household_income` |
| `load_placeholder_311()` | Real San Diego 311 Get It Done complaint counts. Needs: `neighborhood`, `reported_311_complaints` |

As long as the replacement functions return DataFrames with the same column
names, nothing else in the file needs to change — the UI, map, and charts all
read from these three functions.

The `SEVERITY_MAP` dictionary controls the Severe/Light grouping
(Pothole + Alligator Crack = Severe, Longitudinal + Transverse Crack = Light).
Update this if the severity grouping logic changes.

The `compute_equity_score()` function has a placeholder weighting formula
(Severe = 2x, Light = 1x). **This should be finalized with Dr. Fernandez**
before it's used in any real reporting — flagged in the code as a TODO-style
comment.

## Known limitations to keep in mind

- Alligator Crack and Transverse Crack have lower model accuracy (mAP50 ~0.32
  each per the v4 training run), so real Severe/Light counts once plugged in
  will likely be undercounts, particularly for Severe, since Alligator Crack
  feeds into that bucket.
