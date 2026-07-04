"""Distill the rule-based Speed Safety Score into a RandomForestRegressor.

speed_safety_score is a deterministic function of the six FEATURE_COLUMNS
below (see score.calculate_score), so this isn't learning anything the
formula didn't already encode. It's a sanity check that (a) a learned model
recovers the same weighting the Safe System formula uses -- compare
model.feature_importances_ to score.BASE_WEIGHTS in evaluate() -- and (b)
the train/eval plumbing works end to end, ready to point at a real
supervised target (e.g. per-segment crash/fatality data) the moment one
exists. See README.md "Future enhancements" for that gap: the crash data
available so far (ATO Road Safety workbook) is national-level only, one
row per country per year, so it can't be joined to individual segments as
a label without every segment in a country collapsing to the same value.
"""

import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .score import BASE_WEIGHTS

FEATURE_COLUMNS = [
    "speed_gap_norm",
    "road_mismatch",
    "urban_flag",
    "vru_exposure",
    "population_exposure",
    "bio_risk",
    "confidence_weight",
]

TARGET_COLUMN = "speed_safety_score"


def load_training_data(path="outputs/segments_scored.geojson"):
    """Load the scored segments (output of notebook 04) for model development."""
    return gpd.read_file(path)


def train(target_column=TARGET_COLUMN, test_size=0.2, random_state=42):
    """Fit a RandomForestRegressor against target_column, stratified by country.

    Stratifying by country keeps both India and Thailand represented
    proportionally in train and test, rather than a plain random split that
    could (by chance) skew one country into the test set.
    """
    gdf = load_training_data()
    X = gdf[FEATURE_COLUMNS]
    y = gdf[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=gdf["country"]
    )
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model, (X_test, y_test)


def evaluate(model, X_test, y_test):
    """Print held-out R^2/MAE and compare feature_importances_ to score.BASE_WEIGHTS."""
    preds = model.predict(X_test)
    print(f"R^2: {r2_score(y_test, preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")

    print("\nfeature_importances_ vs. score.BASE_WEIGHTS:")
    for col, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
        base_weight = BASE_WEIGHTS.get(col)
        weight_str = f"{base_weight:.2f}" if base_weight is not None else "n/a"
        print(f"  {col:20s} importance={importance:.3f}  base_weight={weight_str}")


if __name__ == "__main__":
    model, (X_test, y_test) = train()
    evaluate(model, X_test, y_test)
