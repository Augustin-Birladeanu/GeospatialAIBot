"""Feature engineering for the Speed Safety Score: gap, mismatch, exposure, risk."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import METRIC_CRS, assign_zone_attribute

# Safe System posted-speed ranges (km/h) by Overture road class.
SAFE_SYSTEM_SPEED_RANGES = {
    "motorway": (80, 110),
    "trunk": (60, 90),
    "primary": (50, 70),
    "secondary": (40, 60),
    "tertiary": (30, 50),
    "residential": (10, 30),
    "living_street": (5, 20),
    "unclassified": (20, 40),
}

# Absolute anchors that map raw km/h quantities onto 0-1 component scales.
# Fixed anchors (rather than dataset-relative min-max scaling) keep a given
# score meaning the same thing in every country the pipeline is applied to,
# and stop a single extreme outlier from compressing everyone else's score.
SPEED_GAP_FULL_SIGNAL_KMH = 20  # driving 20+ km/h over the limit = full-strength signal
ROAD_MISMATCH_FULL_SIGNAL_KMH = 30  # posted 30+ km/h above the class ceiling = full-strength signal

# Fatality-probability lookup for a pedestrian/cyclist struck at a given speed.
BIO_RISK_BANDS = [
    (20, 0.05),
    (30, 0.10),
    (40, 0.30),
    (50, 0.80),
    (60, 0.90),
]
BIO_RISK_ABOVE_60 = 1.00


def compute_speed_gap(gdf):
    """Add speed_gap (F85 minus limit, clipped at 0) and its 0-1 anchored version.

    speed_gap_norm saturates at SPEED_GAP_FULL_SIGNAL_KMH: a segment where
    traffic runs 20+ km/h over the limit gets the full signal (1.0) no
    matter what the worst segment in the dataset does. The previous min-max
    scaling let one 88 km/h outlier compress a serious 20 km/h gap down to
    0.23, and made scores non-comparable across datasets.
    """
    gap = gdf["F85thPercentileSpeed"] - gdf["SpeedLimit"]
    gdf["speed_gap"] = gap.clip(lower=0)
    gdf["speed_gap_norm"] = (gdf["speed_gap"].fillna(0) / SPEED_GAP_FULL_SIGNAL_KMH).clip(upper=1.0)
    return gdf


def compute_road_mismatch(gdf):
    """Add road_mismatch: km/h the posted limit exceeds the class's Safe System max, on a 0-1 anchored scale."""
    class_max = gdf["road_class"].map(lambda c: SAFE_SYSTEM_SPEED_RANGES.get(c, (np.nan, np.nan))[1])
    # A road is only "mismatched" when the posted limit is strictly above the
    # class ceiling. A limit below the ceiling (e.g. 55 km/h on a secondary
    # road capped at 60) is compliant and scores 0, not a positive value.
    # The over-posting is scaled against a fixed 30 km/h anchor rather than
    # the class ceiling itself: dividing by class_max capped the achievable
    # value at ~0.5 for every real over-posting in the data, silently
    # halving this component's weight in the final score.
    over = (gdf["SpeedLimit"] - class_max).clip(lower=0)
    gdf["road_mismatch"] = (over / ROAD_MISMATCH_FULL_SIGNAL_KMH).clip(upper=1.0)
    return gdf


def compute_urban_flag(gdf):
    """Add urban_flag: 1.0 where UrbanPC > 0.5, else 0.0."""
    gdf["urban_flag"] = (gdf["UrbanPC"] > 0.5).astype(float)
    return gdf


def compute_vru_exposure(gdf, helmet_layers=None):
    """Add vru_exposure: 0.40 urban_flag + 0.60 normalised low-helmet-compliance risk by zone.

    helmet_layers: optional dict of {country: zones_geodataframe} with a
    helmet_rate column (0-1). When supplied, the inverse of the local helmet
    wearing rate (i.e. exposure of unprotected riders) is spatially joined to
    each segment and blended with urban_flag. Resolution differs sharply
    between countries (4 zones for Maharashtra vs 77 provinces for Thailand),
    so this is a coarse proxy, not a precise exposure model. With no helmet
    layer supplied, urban_flag alone is used as the proxy.
    """
    if not helmet_layers:
        gdf["vru_exposure"] = gdf["urban_flag"]
        return gdf

    low_helmet_risk = pd.Series(np.nan, index=gdf.index)
    for country, zones in helmet_layers.items():
        country_mask = gdf["country"] == country
        if not country_mask.any():
            continue
        helmet_rate = assign_zone_attribute(gdf.loc[country_mask], zones, METRIC_CRS[country])
        low_helmet_risk.loc[country_mask] = 1 - helmet_rate

    scaler = MinMaxScaler()
    low_helmet_risk_norm = pd.Series(
        scaler.fit_transform(low_helmet_risk.fillna(low_helmet_risk.mean()).to_frame()).flatten(),
        index=gdf.index,
    )
    gdf["vru_exposure"] = (0.40 * gdf["urban_flag"] + 0.60 * low_helmet_risk_norm).clip(0, 1)
    return gdf


def compute_recommended_speed_limit(gdf):
    """Add recommended_speed_limit and speed_limit_gap for each segment.

    recommended_speed_limit interpolates within the segment's road-class
    Safe System range (SAFE_SYSTEM_SPEED_RANGES), pulled toward the class
    minimum as vru_exposure rises toward 1 and toward the class maximum as
    it falls toward 0, rounded to the nearest 10 km/h (how limits are
    actually posted). Segments whose road_class has no defined range get
    NaN. speed_limit_gap is the posted SpeedLimit minus this recommendation
    -- positive means the posted limit is above what Safe System principles
    suggest for that segment's exposure.
    """
    ranges = gdf["road_class"].map(SAFE_SYSTEM_SPEED_RANGES)
    class_min = ranges.map(lambda r: r[0] if isinstance(r, tuple) else np.nan)
    class_max = ranges.map(lambda r: r[1] if isinstance(r, tuple) else np.nan)
    recommended = class_min + (class_max - class_min) * (1 - gdf["vru_exposure"])
    gdf["recommended_speed_limit"] = (recommended / 10).round() * 10
    gdf["speed_limit_gap"] = gdf["SpeedLimit"] - gdf["recommended_speed_limit"]
    return gdf


def _bio_risk_for_speed(speed):
    """Look up the fatality-probability band for a single posted speed limit."""
    if pd.isna(speed):
        return np.nan
    for ceiling, prob in BIO_RISK_BANDS:
        if speed <= ceiling:
            return prob
    return BIO_RISK_ABOVE_60


def compute_bio_risk(gdf):
    """Add bio_risk: fatality probability at the exposure speed, scaled by vru_exposure.

    The exposure speed is max(F85thPercentileSpeed, SpeedLimit) -- the speed
    vulnerable road users are actually exposed to. Using the posted limit
    alone understates the danger where real traffic runs well above it, and
    using F85 alone misses limits that permit lethal speeds on roads where
    traffic happens to be slow today. Multiplying by vru_exposure keeps the
    component targeted at "lethal speeds where unprotected people are":
    a fast rural motorway with nobody on foot is not a VRU risk.
    """
    exposure_speed = pd.concat(
        [gdf["F85thPercentileSpeed"], gdf["SpeedLimit"]], axis=1
    ).max(axis=1)
    fatality_prob = exposure_speed.map(_bio_risk_for_speed)
    gdf["bio_risk"] = fatality_prob * gdf["vru_exposure"]
    return gdf
def compute_confidence_weight(gdf):
    """Add confidence_weight: 0.5 for segments longer than 10km, else 1.0.

    This is a data-confidence indicator (the speed sample may not represent
    a very long segment), NOT a risk discount -- it no longer multiplies the
    Speed Safety Score. Halving the score of long segments made genuinely
    dangerous long roads invisible by conflating uncertainty with safety.
    """
    gdf["confidence_weight"] = np.where(gdf["RoadLength_km"] > 10, 0.5, 1.0)
    return gdf


def compute_mapillary_url(gdf):
    """Add mapillary_url built from the centre point of the StreetImageLink bounding coordinates."""

    def _to_url(value):
        if pd.isna(value):
            return None
        try:
            lon1, lat1, lon2, lat2 = (float(v) for v in str(value).split(","))
        except ValueError:
            return None
        center_lon = (lon1 + lon2) / 2
        center_lat = (lat1 + lat2) / 2
        return f"https://www.mapillary.com/app/?lat={center_lat}&lng={center_lon}&z=16"

    gdf["mapillary_url"] = gdf["StreetImageLink"].map(_to_url)
    return gdf


def engineer_features(gdf, helmet_layers=None, population_rasters=None):
    """Run all feature engineering steps on a reliable-segments GeoDataFrame, in order."""
    if population_rasters:
        from .population import add_population_data

        gdf = add_population_data(gdf, population_rasters)
    elif "population_exposure" not in gdf.columns:
        gdf["population_mean"] = np.nan
        gdf["population_max"] = np.nan
        gdf["population_exposure"] = 0.0
    gdf = compute_speed_gap(gdf)
    gdf = compute_road_mismatch(gdf)
    gdf = compute_urban_flag(gdf)
    gdf = compute_vru_exposure(gdf, helmet_layers)
    gdf = compute_recommended_speed_limit(gdf)
    gdf = compute_bio_risk(gdf)
    gdf = compute_confidence_weight(gdf)
    gdf = compute_mapillary_url(gdf)
    return gdf
