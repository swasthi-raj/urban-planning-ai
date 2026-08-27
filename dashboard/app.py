"""
AI-Powered Urban Infrastructure Monitoring — Dashboard Shell
MOC-LLAB / SDSU

STATUS: Shell built with PLACEHOLDER DATA.
Swap out the functions in the "DATA LAYER" section below once real
model inference results, 311 data, and Census ACS data are ready.
Everything below that section (UI, charts, map) reads from those
functions and does not need to change when real data is plugged in,
as long as the returned DataFrame shapes stay the same.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Urban Infrastructure Equity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# DATA LAYER  —  replace these functions with real data once available
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

# Per-class mAP50 from the v4 (6-class) model — used to weight placeholder
# detection confidence, replace with real per-image inference once run.
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
def load_placeholder_detections(seed: int = 42) -> pd.DataFrame:
    """
    PLACEHOLDER — replace with real YOLOv8-seg inference output.
    Expected real replacement: one row per detected instance, with at
    minimum: neighborhood, class_name, confidence, lat, lon, image_id.
    Underserved neighborhoods are seeded with more deficiency counts
    on purpose here, just to make the placeholder map/charts legible;
    this is NOT a real finding.
    """
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
def load_placeholder_socioeconomic() -> pd.DataFrame:
    """
    PLACEHOLDER — replace with real US Census ACS pull (median household
    income, population density, etc.) joined by neighborhood.
    """
    rng = np.random.default_rng(7)
    data = []
    for name, (_, _, category) in NEIGHBORHOODS.items():
        income = rng.integers(38000, 62000) if category == "Underserved" else rng.integers(95000, 190000)
        data.append({"neighborhood": name, "category": category, "median_household_income": income})
    return pd.DataFrame(data)


@st.cache_data
def load_placeholder_311() -> pd.DataFrame:
    """
    PLACEHOLDER — replace with real San Diego 311 Get It Done complaint
    counts by neighborhood, filtered to relevant infrastructure categories.
    """
    rng = np.random.default_rng(11)
    data = []
    for name, (_, _, category) in NEIGHBORHOODS.items():
        # Placeholder assumption for H5 exploration: underserved areas
        # under-report relative to AI-detected issues. Replace with real counts.
        count = rng.integers(15, 45) if category == "Underserved" else rng.integers(30, 70)
        data.append({"neighborhood": name, "reported_311_complaints": count})
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
    "detections, cross-referenced with 311 complaints and Census ACS data."
)
st.warning(
    "⚠️ Placeholder data shown throughout. Charts and map will update automatically "
    "once real model inference and Census/311 data are connected — no layout changes needed.",
    icon="⚠️",
)

detections = load_placeholder_detections()
socio = load_placeholder_socioeconomic()
complaints_311 = load_placeholder_311()
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
        fig_map = px.scatter_mapbox(
            filtered_equity,
            lat="lat",
            lon="lon",
            size="equity_score",
            color="category",
            hover_name="neighborhood",
            hover_data={"total_detections": True, "severe_count": True, "light_count": True, "lat": False, "lon": False},
            zoom=9.2,
            height=480,
            color_discrete_map={"Underserved": "#d62728", "Wealthier": "#1f77b4"},
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
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
st.subheader("Neighborhood equity summary")
display_table = filtered_equity.merge(socio, on=["neighborhood", "category"], how="left").merge(
    complaints_311, on="neighborhood", how="left"
)
display_table = display_table[
    ["neighborhood", "category", "total_detections", "severe_count", "light_count",
     "equity_score", "median_household_income", "reported_311_complaints"]
].rename(columns={
    "neighborhood": "Neighborhood", "category": "Type", "total_detections": "Total Detections",
    "severe_count": "Severe", "light_count": "Light", "equity_score": "Equity Score",
    "median_household_income": "Median Income ($)", "reported_311_complaints": "311 Complaints",
})
st.dataframe(display_table, use_container_width=True, hide_index=True)

st.divider()

# ---- H2 exploration: income vs deficiency scatter ----
st.subheader("Equity check: income vs. infrastructure deficiency (H2)")
scatter_data = filtered_equity.merge(socio, on=["neighborhood", "category"], how="left")
if len(scatter_data):
    fig_scatter = px.scatter(
        scatter_data, x="median_household_income", y="equity_score",
        color="category", size="total_detections", hover_name="neighborhood",
        color_discrete_map={"Underserved": "#d62728", "Wealthier": "#1f77b4"},
        labels={"median_household_income": "Median Household Income ($)", "equity_score": "Equity Score (weighted deficiencies)"},
        height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "Placeholder data is seeded to show a downward trend on purpose for layout testing. "
        "This is not a real finding — re-run once actual inference and ACS data are connected."
    )

# ---- H5 exploration: AI detections vs 311 reports ----
st.subheader("AI detections vs. citizen 311 reports (H5)")
h5_data = filtered_equity.merge(complaints_311, on="neighborhood", how="left")
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
    "Data engine: Google Street View + YOLOv8-seg (v4, 6-class) · San Diego 311 Get It Done · "
    "US Census ACS · MOC-LLAB, SDSU"
)
