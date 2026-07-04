"""Backfill sparse road names using Overture Maps' own road-name data.

The Speed Safety Score's road_name field (harmonize_schema) only carries
whatever the challenge's raw exports populated -- names_primary for India,
english_ro for Thailand -- and both are sparsely filled: ~29% of reliable
India segments and ~31% of reliable Thailand segments have a real name, the
rest get a generated "{road_class} segment {segment_id}" label. A live check
against Overture Maps' current road data for a Maharashtra sample found only
~34% name coverage there too, confirming this is a genuine OpenStreetMap/
Overture road-naming completeness gap for these regions' minor and rural
roads -- not an artifact of this specific export -- so this backfill is a
real but modest improvement, not a full fix.

Fetching Overture's road segments for a whole state/country isn't cheap
enough to do inline (a naive full download is tens of GB), so
scratchpad/fetch_overture_names.py fetches only what this needs -- named,
subtype=road, our used road classes -- server-side filtered, and this module
does the local spatial match against that pre-fetched file.
"""

import geopandas as gpd
import pandas as pd
from shapely import wkb

# Road centerlines from two different sources (this project's segments vs.
# Overture's) rarely align exactly, so the match uses a small tolerance
# rather than requiring the geometries to touch.
MAX_MATCH_DISTANCE_M = 50


def load_overture_names(parquet_path):
    """Load a fetch_overture_names.py output into a name-only line GeoDataFrame."""
    df = pd.read_parquet(parquet_path)
    df["geometry"] = df["geometry"].map(wkb.loads)
    df["name"] = df["names"].map(lambda n: n.get("primary") if isinstance(n, dict) else None)
    gdf = gpd.GeoDataFrame(df[["name", "class", "geometry"]], geometry="geometry", crs="EPSG:4326")
    return gdf[gdf["name"].notna() & (gdf["name"].str.strip() != "")]


def backfill_road_names(segments, overture_names, metric_crs, max_distance_m=MAX_MATCH_DISTANCE_M):
    """Fill road_name for segments flagged road_name_is_fallback, in place of the generated label.

    Only attempts segments where road_name_is_fallback is True (segments that
    already have a name from the raw export are left untouched). Matches each
    such segment's representative point against the nearest named Overture
    segment within max_distance_m; segments with nothing that close keep
    their existing fallback label.
    """
    to_fill_mask = segments["road_name_is_fallback"]
    if not to_fill_mask.any() or overture_names.empty:
        return segments

    points = segments.loc[to_fill_mask, ["geometry"]].copy()
    points["geometry"] = points.geometry.representative_point()
    points = points.to_crs(metric_crs)
    lines = overture_names[["name", "geometry"]].to_crs(metric_crs)

    matched = gpd.sjoin_nearest(points, lines, how="left", max_distance=max_distance_m, distance_col="match_distance_m")
    matched = matched[~matched.index.duplicated(keep="first")]

    out = segments.copy()
    filled = matched["name"].reindex(out.index)
    backfilled_mask = to_fill_mask & filled.notna()
    out.loc[backfilled_mask, "road_name"] = filled.loc[backfilled_mask]
    out.loc[backfilled_mask, "road_name_is_fallback"] = False
    return out
