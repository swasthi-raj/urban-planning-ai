"""
Shared configuration for the MOC-LLAB urban infrastructure pipeline.

Central place for the 7 study neighborhoods, their coordinates, and the
external data source identifiers (San Diego Get It Done comm_plan_name,
Street View sampling radius, etc.) used by process_311.py and
download_gsv.py.
"""

import os

# =========================================================================
# Study neighborhoods
# =========================================================================
# comm_plan_name must match the `comm_plan_name` field in the City of San
# Diego "Get It Done" 311 dataset (https://data.sandiego.gov/datasets/get-it-done-311/)
# so process_311.py can filter reports to these areas.
#
# NOTE — El Cajon is its own incorporated city, NOT a City of San Diego
# community planning area, so it has no comm_plan_name and will not appear
# in San Diego's Get It Done dataset. It's kept in this list for GSV image
# collection and model inference, but process_311.py skips it (comm_plan_name
# is None) per your call to skip 311 cross-validation for that neighborhood only.
NEIGHBORHOODS = {
    "Barrio Logan": {
        "lat": 32.6980, "lon": -117.1440,
        "category": "Underserved",
        "comm_plan_name": "BARRIO LOGAN",
    },
    "City Heights": {
        "lat": 32.7490, "lon": -117.1150,
        "category": "Underserved",
        "comm_plan_name": "CITY HEIGHTS",
    },
    "El Cajon": {
        "lat": 32.7948, "lon": -116.9625,
        "category": "Underserved",
        "comm_plan_name": None,  # separate city — no San Diego 311 coverage
    },
    "Encanto": {
        "lat": 32.7075, "lon": -117.0525,
        "category": "Underserved",
        "comm_plan_name": "ENCANTO NEIGHBORHOODS",
    },
    "La Jolla": {
        "lat": 32.8328, "lon": -117.2713,
        "category": "Wealthier",
        "comm_plan_name": "LA JOLLA",
    },
    "Mission Hills": {
        "lat": 32.7503, "lon": -117.1825,
        "category": "Wealthier",
        "comm_plan_name": "MISSION HILLS",
    },
    "Del Mar Heights": {
        "lat": 32.9595, "lon": -117.2494,
        "category": "Wealthier",
        "comm_plan_name": "DEL MAR HEIGHTS",
    },
}

# =========================================================================
# Infrastructure detection classes (must match the trained YOLOv8-seg model)
# =========================================================================
CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
    "Side_Walk",
    "Bike_Lane",
]

SEVERITY_MAP = {
    "Pothole": "Severe",
    "Alligator Crack": "Severe",
    "Longitudinal Crack": "Light",
    "Transverse Crack": "Light",
    "Side_Walk": "N/A",
    "Bike_Lane": "N/A",
}

# Keywords used to filter Get It Done 311 reports down to infrastructure
# categories comparable to the YOLOv8 detection classes above. Matched
# case-insensitively against service_name + service_name_detail.
INFRASTRUCTURE_311_KEYWORDS = [
    "pothole", "pavement", "crack", "sidewalk", "walkway",
    "bike lane", "bicycle", "road", "street failure", "curb",
]

# =========================================================================
# Google Street View
# =========================================================================
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
GSV_IMAGE_SIZE = "640x640"
GSV_SAMPLES_PER_NEIGHBORHOOD = 40          # grid-sampled points per neighborhood
GSV_NEIGHBORHOOD_RADIUS_METERS = 1200      # sampling radius around each center point
GSV_TARGETED_311_LIMIT = 40                # extra images pulled at 311 complaint locations

# =========================================================================
# Paths
# =========================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_311_DIR = os.path.join(DATA_DIR, "311_raw")
IMAGES_DIR = os.path.join(DATA_DIR, "gsv_images")
DETECTIONS_CSV = os.path.join(DATA_DIR, "detections.csv")
COMPLAINTS_311_CSV = os.path.join(DATA_DIR, "complaints_311.csv")
