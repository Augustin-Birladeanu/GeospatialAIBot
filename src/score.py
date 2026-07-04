"""Speed Safety Score: weighted blend of risk features, risk tiers, and sensitivity testing."""

from itertools import product

import numpy as np
import pandas as pd

# Population receives 15% of the score. The pre-population model's three
# components retain their relative emphasis within the remaining 85%.
BASE_WEIGHTS = {
    "speed_gap_norm": 0.25,
    "road_mismatch": 0.25,
    "bio_risk": 0.35,
    "population_exposure": 0.15,
}

RISK_TIER_THRESHOLDS = [(70, "High risk"), (40, "Medium risk")]
INSUFFICIENT_DATA_TIER = "Insufficient data"


def calculate_score(gdf, weights=BASE_WEIGHTS):
    """Return the 0-100 speed_safety_score series for a given set of component weights."""
    weighted_sum = sum(gdf[col] * w for col, w in weights.items())
    return (weighted_sum * 100).round(1)


def assign_risk_tier(score):
    """Classify a speed_safety_score series into High/Medium/Low risk tiers."""
    # Apply in ascending-threshold order last-wins, so High risk overrides Medium.
    tier = pd.Series("Low risk", index=score.index)
    tier = tier.mask(score >= RISK_TIER_THRESHOLDS[1][0], RISK_TIER_THRESHOLDS[1][1])
    tier = tier.mask(score >= RISK_TIER_THRESHOLDS[0][0], RISK_TIER_THRESHOLDS[0][1])
    return tier


def score_and_classify(gdf, weights=BASE_WEIGHTS):
    """Add speed_safety_score and risk_tier columns to a feature-engineered GeoDataFrame."""
    gdf["speed_safety_score"] = calculate_score(gdf, weights)
    gdf["risk_tier"] = assign_risk_tier(gdf["speed_safety_score"])
    population_weight = weights.get("population_exposure", 0)
    gdf["population_score_contribution"] = (
        gdf["population_exposure"] * population_weight * 100
    ).round(1)
    gdf["score_without_population"] = (
        gdf["speed_safety_score"] - gdf["population_score_contribution"]
    ).clip(lower=0).round(1)
    gdf["risk_tier_without_population"] = assign_risk_tier(gdf["score_without_population"])
    return gdf


def _weight_variants(base_weights, step=0.05, max_delta=0.10):
    """Yield every weight combination within +-max_delta of base_weights that still sums to 1.0."""
    keys = list(base_weights.keys())
    deltas = np.round(np.arange(-max_delta, max_delta + step / 2, step), 2)
    for combo in product(deltas, repeat=len(keys)):
        if abs(sum(combo)) > 1e-9:
            continue
        candidate = {k: round(base_weights[k] + d, 2) for k, d in zip(keys, combo)}
        if any(v < 0 for v in candidate.values()):
            continue
        yield candidate


def run_sensitivity_analysis(gdf, base_weights=BASE_WEIGHTS, top_pct=0.20, step=0.05, max_delta=0.10):
    """Vary each weight +-max_delta and measure top-N% overlap against the baseline scoring."""
    n_top = max(1, int(round(len(gdf) * top_pct)))

    baseline_score = calculate_score(gdf, base_weights)
    baseline_top = set(baseline_score.nlargest(n_top).index)

    overlaps = []
    flip_counts = pd.Series(0, index=gdf.index)
    n_variants = 0
    for weights in _weight_variants(base_weights, step=step, max_delta=max_delta):
        if weights == base_weights:
            continue
        n_variants += 1
        variant_score = calculate_score(gdf, weights)
        variant_top = set(variant_score.nlargest(n_top).index)
        overlap_pct = len(baseline_top & variant_top) / n_top * 100
        overlaps.append(overlap_pct)
        flipped = baseline_top.symmetric_difference(variant_top)
        flip_counts.loc[list(flipped)] += 1

    avg_overlap = float(np.mean(overlaps)) if overlaps else 100.0
    return {
        "average_overlap_pct": avg_overlap,
        "n_variants": n_variants,
        "flip_counts": flip_counts[flip_counts > 0].sort_values(ascending=False),
        "robust": avg_overlap > 70,
    }
