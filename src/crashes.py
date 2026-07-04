"""Join point-level Thailand crash reports onto road segments for score validation.

Thailandaccident2025.xlsx (Department of Highways accident log) is the first
segment-joinable crash data this project has had access to -- the ATO Road
Safety workbook referenced in src/train_model.py is national-level only, one
row per country per year. This is genuine point-level ground truth: 2025
crash records with lat/lon, fatality, and injury counts.

It is used for VALIDATION and map enrichment, not folded into the weighted
Speed Safety Score itself: the score's components are deliberately built on
absolute anchors so a given score means the same thing in India and
Thailand (see docs/methodology.md). Crash records exist for Thailand only,
so adding them as a scoring term would make the Thailand formula structurally
different from India's -- undermining that cross-country consistency. Instead
this module answers "does the score actually track where crashes happen?"
"""

import geopandas as gpd
import pandas as pd

from .utils import COUNTRY_BOUNDS, METRIC_CRS

FATALITY_COL = "ผู้เสียชีวิต"
SERIOUS_INJURY_COL = "ผู้บาดเจ็บสาหัส"
MINOR_INJURY_COL = "ผู้บาดเจ็บเล็กน้อย"

# A crash point is matched to the nearest segment only within this radius --
# beyond it the nearest segment is more likely a mismatch (wrong carriageway,
# sparse rural network) than the true crash location.
MAX_MATCH_DISTANCE_M = 300


def load_crash_data(path):
    """Load the raw Thailand crash workbook into a point GeoDataFrame in EPSG:4326.

    Drops records with missing coordinates or coordinates outside Thailand's
    bounding box (a handful of rows have swapped or corrupted lat/lon --
    e.g. a LATITUDE of 100+ -- which are unrecoverable without the source
    report, so they are excluded rather than guessed at).
    """
    df = pd.read_excel(path)
    n_total = len(df)

    bounds = COUNTRY_BOUNDS["Thailand"]
    valid = (
        df["LATITUDE"].between(*bounds["lat"])
        & df["LONGITUDE"].between(*bounds["lon"])
    )
    dropped = n_total - int(valid.sum())
    df = df[valid].copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]),
        crs="EPSG:4326",
    )
    gdf.attrs["n_total"] = n_total
    gdf.attrs["n_dropped_invalid_coords"] = dropped
    return gdf


def match_crashes_to_segments(crashes, segments, max_distance_m=MAX_MATCH_DISTANCE_M):
    """Nearest-join each crash point to a Thailand road segment within max_distance_m.

    Returns crashes with a segment_id column (NaN where nothing is within range).
    """
    metric_crs = METRIC_CRS["Thailand"]
    crashes_m = crashes.to_crs(metric_crs)
    segments_m = segments[["segment_id", "geometry"]].to_crs(metric_crs)

    joined = gpd.sjoin_nearest(
        crashes_m, segments_m, how="left", max_distance=max_distance_m, distance_col="match_distance_m"
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined


def aggregate_crashes_to_segments(crashes, segments, max_distance_m=MAX_MATCH_DISTANCE_M):
    """Match crashes to segments and aggregate crash/fatality/injury counts per segment_id.

    Returns a DataFrame indexed by segment_id with crash_count, fatality_count,
    serious_injury_count, minor_injury_count, ksi_count (killed or seriously
    injured -- the standard road-safety severity metric).
    """
    matched = match_crashes_to_segments(crashes, segments, max_distance_m)
    matched_hit = matched[matched["segment_id"].notna()]

    agg = matched_hit.groupby("segment_id").agg(
        crash_count=("segment_id", "size"),
        fatality_count=(FATALITY_COL, "sum"),
        serious_injury_count=(SERIOUS_INJURY_COL, "sum"),
        minor_injury_count=(MINOR_INJURY_COL, "sum"),
    )
    agg["ksi_count"] = agg["fatality_count"] + agg["serious_injury_count"]

    agg.attrs["n_crashes_input"] = len(crashes)
    agg.attrs["n_crashes_matched"] = len(matched_hit)
    agg.attrs["match_rate_pct"] = round(len(matched_hit) / len(crashes) * 100, 1) if len(crashes) else 0.0
    return agg


def attach_crash_counts(gdf, crash_agg):
    """Left-join crash aggregates onto a segments GeoDataFrame, filling 0 for no-match segments.

    Segments outside Thailand (no crash data available for their country) are
    NOT filled with 0 -- they get NaN so it's clear no crash data exists for
    that country, as opposed to Thailand segments with zero matched crashes.
    """
    out = gdf.merge(crash_agg.reset_index(), on="segment_id", how="left")
    count_cols = ["crash_count", "fatality_count", "serious_injury_count", "minor_injury_count", "ksi_count"]
    is_thailand = out["country"] == "Thailand"
    for col in count_cols:
        out.loc[is_thailand, col] = out.loc[is_thailand, col].fillna(0)
    return out
