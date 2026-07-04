"""SCAFFOLD ONLY -- intentionally not executed by the automated pipeline.

The Speed Safety Score in score.py is a deterministic, hand-weighted formula,
not a trained model. This file is a starting point if/when the team wants to
fit an actual predictive model (e.g. to learn feature weights from outcome
data, or to validate the rule-based score against an external target such as
RankedPercentile or real crash records). Nothing here is run as part of
notebooks/scripts in this repo -- fill in a real target and run it yourselves.

Suggested next steps, left undone on purpose:
  1. Decide the supervised target (RankedPercentile? observed crash/fatality
     data, if it becomes available? a hand-labelled subset?).
  2. Choose train/test split strategy -- likely by country and/or road class
     to avoid leakage between geographically adjacent segments.
  3. Try a baseline (e.g. RandomForestRegressor) against the engineered
     features in segments_scored.geojson, and compare its ranking of
     segments to the rule-based speed_safety_score.
  4. Validate carefully before trusting it over the transparent rule-based
     score -- the Safe System weighting is the explainable baseline this
     challenge submission relies on.
"""

import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "speed_gap_norm",
    "road_mismatch",
    "urban_flag",
    "vru_exposure",
    "population_exposure",
    "bio_risk",
    "confidence_weight",
]

TARGET_COLUMN = None  
# The target column is the outcome the model will learn to predict.
# Replace None with a valid column name (e.g. "RankedPercentile") before training.
# TODO: set this once a real supervised target is agreed. 


def load_training_data(path="outputs/segments_scored.geojson"):
    """Load the scored segments for model development; does not fit anything."""
    return gpd.read_file(path)


def train(target_column=TARGET_COLUMN, test_size=0.2, random_state=42):
    """Placeholder training routine -- raises until a target column is supplied."""
    if target_column is None:
        raise NotImplementedError(
            "Set TARGET_COLUMN (or pass target_column=...) before running this. "
            "This function is intentionally left unfinished -- see module docstring."
        )
    gdf = load_training_data()
    X = gdf[FEATURE_COLUMNS]
    y = gdf[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model, (X_test, y_test)


if __name__ == "__main__":
    raise SystemExit(
        "This script is a scaffold and is not meant to be run automatically. "
        "Set TARGET_COLUMN and call train() yourself when ready."
    )
