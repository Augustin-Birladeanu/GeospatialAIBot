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

# Fatality-probability lookup for a pedestrian/cyclist struck at a given posted speed.
BIO_RISK_BANDS = [
    (20, 0.05),
    (30, 0.10),
    (40, 0.30),
    (50, 0.80),
    (60, 0.90),
]
BIO_RISK_ABOVE_60 = 1.00


def compute_speed_gap(gdf):
    """Add speed_gap (F85 minus limit, clipped at 0) and its 0-1 normalised version."""
    gap = gdf["F85thPercentileSpeed"] - gdf["SpeedLimit"]
    gdf["speed_gap"] = gap.clip(lower=0)
    scaler = MinMaxScaler()
    gdf["speed_gap_norm"] = scaler.fit_transform(gdf[["speed_gap"]].fillna(0))
    return gdf


def compute_road_mismatch(gdf):
    """Add road_mismatch: how far the posted limit exceeds the class's Safe System max, clipped at 1."""
    class_max = gdf["road_class"].map(lambda c: SAFE_SYSTEM_SPEED_RANGES.get(c, (np.nan, np.nan))[1])
    # A road is only "mismatched" when the posted limit is strictly above the
    # class ceiling. A limit below the ceiling (e.g. 55 km/h on a secondary
    # road capped at 60) is compliant and scores 0, not a positive value.
    over = (gdf["SpeedLimit"] - class_max).clip(lower=0)
    gdf["road_mismatch"] = (over / class_max).clip(upper=1.0)
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


def _bio_risk_for_speed(speed):
    """Look up the fatality-probability band for a single posted speed limit."""
    if pd.isna(speed):
        return np.nan
    for ceiling, prob in BIO_RISK_BANDS:
        if speed <= ceiling:
            return prob
    return BIO_RISK_ABOVE_60


def compute_bio_risk(gdf):
    """Add bio_risk: speed-based fatality probability multiplied by vru_exposure."""
    fatality_prob = gdf["SpeedLimit"].map(_bio_risk_for_speed)
    gdf["bio_risk"] = fatality_prob * gdf["vru_exposure"]
    return gdf


def compute_confidence_weight(gdf):
    """Add confidence_weight: 0.5 for segments longer than 10km, else 1.0."""
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
    gdf = compute_bio_risk(gdf)
    gdf = compute_confidence_weight(gdf)
    gdf = compute_mapillary_url(gdf)
    return gdf
