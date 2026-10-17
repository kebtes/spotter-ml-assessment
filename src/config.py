from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "train-test.csv"
VAL_PATH = DATA_DIR / "validation.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"

OUTPUT_DIR = BASE_DIR / "scorer_results"
PREDICTIONS_OUTPUT_PATH = BASE_DIR / "validation_predictions.csv"
DECEMBER_OUTPUT_PATH = DECEMBER_PATH

# Modeling Constants
RANDOM_STATE = 42
EARTH_RADIUS_MILES = 3958.8

# Fixed Lexington to Fort Wayne lane defaults (for December chart inputs)
LEXINGTON_COORDS = (36.99152, -84.99876)
FORT_WAYNE_COORDS = (41.31561, -85.36206)

# Feature sets
CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery", "lane"]

NUMERIC_FEATURES = [
    "distance", "haversine", "circuity", "lat_diff", "lon_diff",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "weight_clean", "weight_per_mile", "is_heavy",
    "dayofweek", "is_weekend", "day", "month", "dayofyear", "days_to_month_end",
    "days_to_thanksgiving", "days_to_christmas", "near_christmas", "pre_christmas_rush", "post_christmas",
    "base_cost_estimate", "mkt_index_clean", "mkt_x_quote", "quote_per_mkt", "quote_signal"
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
