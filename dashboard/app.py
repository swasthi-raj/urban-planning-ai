"""
AI-Powered Urban Infrastructure Monitoring — Dashboard
MOC-LLAB / SDSU

Reads real YOLOv8-seg detections (data/detections.csv) when present —
produced by run_inference.py against the trained model (best (3).pt) on
geotagged San Diego street-view imagery. Falls back to placeholder data
so the dashboard still runs before that file exists.

311 complaint counts come from data/complaints_311.csv (produced by
process_311.py against San Diego's Get It Done open dataset) when present,
otherwise placeholder counts are used the same way.
"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Urban Infrastructure Equity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DETECTIONS_CSV = os.path.join(DATA_DIR, "detections.csv")
COMPLAINTS_311_CSV = os.path.join(DATA_DIR, "complaints_311.csv")

# =========================================================================
# DATA LAYER
# =========================================================================

NEIGHBORHOODS = {
    # name: (lat, lon, category)
    "Barrio Logan":    (32.6980, -117.1440, "Underserved"),
    "City Heights":    (32.7490, -117.1150, "Underserved"),
    "El Cajon":        (32.7948, -116.9625, "Underserved"),
    "Encanto":         (32.7075, -117.0525, "Underserved"),
    "La Jolla":        (32.8328, -117.2713, "Wealthier"),
    "Mission Hills":   (32.7503, -117.1825, "Wealthier"),
    "Del Mar Heights": (32.9595, -117.2494, "Wealthier"),
}

# Real per-class mAP50 from the trained model (best (3).pt, yolov8m-seg,
# 60 epochs, overall mAP50 = 0.577) — not a placeholder, this is the actual
# training result read from training_results (3).zip / results.csv.
CLASS_MODEL_CONFIDENCE = {
    "Longitudinal Crack": 0.550,
    "Transverse Crack": 0.318,
    "Alligator Crack": 0.328,
    "Pothole": 0.501,
    "Side_Walk": 0.847,
    "Bike_Lane": 0.921,
}

SEVERITY_MAP = {
    "Pothole": "Severe",
    "Alligator Crack": "Severe",
    "Longitudinal Crack": "Light",
    "Transverse Crack": "Light",
    "Side_Walk": "N/A",
    "Bike_Lane": "N/A",
}


@st.cache_data
def load_detections(seed: int = 42) -> pd.DataFrame:
    """
    Real detections when data/detections.csv exists (produced by
    run_inference.py). Falls back to seeded placeholder rows otherwise,
    so the dashboard is still runnable before inference has been run.
    """
    if os.path.exists(DETECTIONS_CSV):
        df = pd.read_csv(DETECTIONS_CSV)
        expected = {"neighborhood", "category", "class_name", "severity", "confidence", "lat", "lon"}
        missing = expected - set(df.columns)
        if missing:
            st.error(f"detections.csv is missing expected columns: {missing}")
        return df

    rng = np.random.default_rng(seed)
    rows = []
    classes = list(CLASS_MODEL_CONFIDENCE.keys())
    for name, (lat, lon, category) in NEIGHBORHOODS.items():
        base_count = rng.integers(180, 260) if category == "Underserved" else rng.integers(70, 150)
        for _ in range(base_count):
            cls = rng.choice(classes, p=[0.28, 0.18, 0.08, 0.22, 0.14, 0.10])
            rows.append({
                "neighborhood": name,
                "category": category,
                "class_name": cls,
                "severity": SEVERITY_MAP[cls],
                "confidence": round(float(np.clip(rng.normal(CLASS_MODEL_CONFIDENCE[cls], 0.1), 0.25, 0.98)), 2),
                "lat": lat + rng.normal(0, 0.01),
                "lon": lon + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


@st.cache_data
def load_311() -> pd.DataFrame:
    """
    Real San Diego Get It Done 311 counts when data/complaints_311.csv
    exists (produced by process_311.py). Falls back to placeholder counts
    otherwise. El Cajon has no San Diego 311 coverage (separate city) —
    the real file marks this with source="no_311_coverage"; the placeholder
    path below approximates the same by giving it a count without a
    real source label.
    """
    if os.path.exists(COMPLAINTS_311_CSV):
        df = pd.read_csv(COMPLAINTS_311_CSV)
        if "source" not in df.columns:
            df["source"] = "unknown"
        return df

    rng = np.random.default_rng(11)
    data = []
    for name, (_, _, category) in NEIGHBORHOODS.items():
        count = rng.integers(15, 45) if category == "Underserved" else rng.integers(30, 70)
        data.append({
            "neighborhood": name,
            "reported_311_complaints": count,
            "source": "placeholder",
        })
    return pd.DataFrame(data)


def compute_equity_score(detections: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder equity scoring logic: weights Severe detections higher
    than Light. This formula is illustrative only — the real scoring
    methodology should be finalized with Dr. Fernandez before this
    feeds into any reporting.
    """
    weight = {"Severe": 2.0, "Light": 1.0, "N/A": 0.0}
    d = detections.copy()
    d["weighted_score"] = d["severity"].map(weight)
    summary = (
        d.groupby(["neighborhood", "category"])
        .agg(
            total_detections=("class_name", "count"),
            severe_count=("severity", lambda s: (s == "Severe").sum()),
            light_count=("severity", lambda s: (s == "Light").sum()),
            equity_score=("weighted_score", "sum"),
        )
        .reset_index()
    )
    summary["lat"] = summary["neighborhood"].map(lambda n: NEIGHBORHOODS[n][0])
    summary["lon"] = summary["neighborhood"].map(lambda n: NEIGHBORHOODS[n][1])
    return summary.sort_values("equity_score", ascending=False)


# =========================================================================
# UI LAYER
# =========================================================================

st.title("AI-Powered Urban Infrastructure Monitoring")
st.caption(
    "Equity scoring across San Diego neighborhoods — Google Street View + YOLOv8-seg "
    "detections, cross-referenced with 311 complaints."
)

detections = load_detections()
complaints_311 = load_311()
using_real_detections = os.path.exists(DETECTIONS_CSV)
using_real_311 = os.path.exists(COMPLAINTS_311_CSV)

if using_real_detections and using_real_311:
    st.success(
        f"✅ Showing real data: {len(detections):,} model detections across "
        f"{detections['neighborhood'].nunique()} neighborhoods, cross-referenced with real "
        "San Diego 311 Get It Done complaints.",
        icon="✅",
    )
else:
    missing = []
    if not using_real_detections:
        missing.append("model detections (run run_inference.py)")
    if not using_real_311:
        missing.append("311 data (run process_311.py)")
    st.warning(
        f"⚠️ Placeholder data shown for: {', '.join(missing)}. Layout and charts will "
        "update automatically once those scripts have been run — no code changes needed.",
        icon="⚠️",
    )

equity = compute_equity_score(detections)

# ---- Sidebar filters ----
st.sidebar.header("Filters")
selected_categories = st.sidebar.multiselect(
    "Neighborhood type", options=["Underserved", "Wealthier"], default=["Underserved", "Wealthier"]
)
selected_classes = st.sidebar.multiselect(
    "Infrastructure class", options=list(CLASS_MODEL_CONFIDENCE.keys()), default=list(CLASS_MODEL_CONFIDENCE.keys())
)
selected_severity = st.sidebar.multiselect(
    "Severity", options=["Severe", "Light", "N/A"], default=["Severe", "Light", "N/A"]
)

filtered = detections[
    detections["category"].isin(selected_categories)
    & detections["class_name"].isin(selected_classes)
    & detections["severity"].isin(selected_severity)
]
filtered_equity = compute_equity_score(filtered) if len(filtered) else equity.iloc[0:0]

# ---- KPI row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total detections", f"{len(filtered):,}")
col2.metric("Severe issues", f"{(filtered['severity'] == 'Severe').sum():,}")
col3.metric("Light issues", f"{(filtered['severity'] == 'Light').sum():,}")
top_neighborhood = filtered_equity.iloc[0]["neighborhood"] if len(filtered_equity) else "—"
col4.metric("Highest equity score", top_neighborhood)

st.divider()

# ---- Map + bar chart ----
map_col, chart_col = st.columns([1.3, 1])

with map_col:
    st.subheader("Detection density by neighborhood")
    if len(filtered_equity):
        fig_map = px.scatter_map(
            filtered_equity,
            lat="lat",
            lon="lon",
            size="equity_score",
            color="category",
            hover_name="neighborhood",
            hover_data={"total_detections": True, "severe_count": True, "light_count": True, "lat": False, "lon": False},
            zoom=9.2,
            height=480,
            map_style="open-street-map",
            color_discrete_map={"Underserved": "#d62728", "Wealthier": "#1f77b4"},
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No detections match the current filters.")

with chart_col:
    st.subheader("Detections by class")
    if len(filtered):
        class_counts = filtered["class_name"].value_counts().reset_index()
        class_counts.columns = ["class_name", "count"]
        fig_bar = px.bar(
            class_counts, x="count", y="class_name", orientation="h",
            color="class_name", height=480,
        )
        fig_bar.update_layout(showlegend=False, yaxis_title="", xaxis_title="Detections")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No detections match the current filters.")

st.divider()

# ---- Neighborhood comparison table ----
# Always show all 7 study neighborhoods, even ones with no real detections
# yet — as an explicit "No data yet" rather than silently omitting the row
# (which reads as "this neighborhood wasn't part of the study") or showing
# 0 (which misleadingly reads as "no issues found").
st.subheader("Neighborhood equity summary")
all_neighborhoods_df = pd.DataFrame([
    {"neighborhood": n, "category": info[2]} for n, info in NEIGHBORHOODS.items()
])
display_table = all_neighborhoods_df.merge(filtered_equity, on=["neighborhood", "category"], how="left")
display_table = display_table.merge(
    complaints_311[["neighborhood", "reported_311_complaints", "source"]], on="neighborhood", how="left"
)
for col in ["total_detections", "severe_count", "light_count", "equity_score"]:
    display_table[col] = display_table.apply(
        lambda r, c=col: "No data yet" if using_real_detections and r["neighborhood"] not in set(detections["neighborhood"]) else (r[c] if pd.notna(r[c]) else 0),
        axis=1,
    )
display_table["reported_311_complaints"] = display_table.apply(
    lambda r: "N/A (no SD 311 coverage)" if r.get("source") == "no_311_coverage" else r["reported_311_complaints"],
    axis=1,
)
display_table = display_table[
    ["neighborhood", "category", "total_detections", "severe_count", "light_count",
     "equity_score", "reported_311_complaints"]
].rename(columns={
    "neighborhood": "Neighborhood", "category": "Type", "total_detections": "Total Detections",
    "severe_count": "Severe", "light_count": "Light", "equity_score": "Equity Score",
    "reported_311_complaints": "311 Complaints",
})
st.dataframe(display_table, use_container_width=True, hide_index=True)
if using_real_detections:
    missing = [n for n in NEIGHBORHOODS if n not in set(detections["neighborhood"])]
    if missing:
        st.caption(
            f"'No data yet' means no real street-view imagery has been collected/run through "
            f"the model for that neighborhood ({', '.join(missing)}) — not that no issues exist. "
            "Run download_gsv.py + run_inference.py to fill these in."
        )

st.divider()

# ---- Sample detections by neighborhood ----
st.subheader("Sample detections by neighborhood")
st.caption("Real model output on real street-view imagery — boxes/labels are the trained model's actual predictions, not mockups.")
SAMPLE_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
sample_cols = st.columns(4)
for i, name in enumerate(NEIGHBORHOODS):
    col = sample_cols[i % 4]
    img_path = os.path.join(SAMPLE_IMAGES_DIR, f"{name.replace(' ', '_')}.jpg")
    with col:
        if os.path.exists(img_path):
            st.image(img_path, caption=name, use_container_width=True)
        else:
            st.info(f"No sample image yet for {name}")

st.divider()

# ---- H5 exploration: AI detections vs 311 reports ----
st.subheader("AI detections vs. citizen 311 reports (H5)")
h5_data = filtered_equity.merge(complaints_311[["neighborhood", "reported_311_complaints", "source"]], on="neighborhood", how="left")
h5_data = h5_data[h5_data["source"] != "no_311_coverage"]  # El Cajon has no SD 311 comparison
if len(h5_data):
    h5_melt = h5_data.melt(
        id_vars=["neighborhood", "category"],
        value_vars=["total_detections", "reported_311_complaints"],
        var_name="source", value_name="count",
    )
    h5_melt["source"] = h5_melt["source"].map({
        "total_detections": "AI-detected", "reported_311_complaints": "311 reported"
    })
    fig_h5 = px.bar(
        h5_melt, x="neighborhood", y="count", color="source", barmode="group", height=420,
    )
    st.plotly_chart(fig_h5, use_container_width=True)
    st.caption(
        "Cross-validates AI-detected infrastructure conditions against citizen-reported "
        "311 complaints. El Cajon is excluded here — it's a separate incorporated city with "
        "no San Diego 311 coverage."
    )

st.caption(
    "Data engine: Google Street View + YOLOv8-seg (yolov8m-seg, 6-class, mAP50 0.577) · "
    "San Diego 311 Get It Done · MOC-LLAB, SDSU"
)
