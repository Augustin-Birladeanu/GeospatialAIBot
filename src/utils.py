"""Loading, schema harmonisation, country detection, and filtering helpers."""

import geopandas as gpd
import pandas as pd

# Approximate coordinate bounds used to tell India and Thailand segments apart.
COUNTRY_BOUNDS = {
    "India": {"lon": (68, 97), "lat": (8, 37)},
    "Thailand": {"lon": (97, 106), "lat": (5, 21)},
}

# Metric (projected) CRS used for any distance-based calculation, by country.
METRIC_CRS = {"India": "EPSG:32643", "Thailand": "EPSG:32647"}

# The two source files use different field names for the same concepts
# (Maharashtra/India matches the challenge field list, Thailand matches the
# Agilysis data-guide PDF). These map each raw schema onto a common set of
# columns used by the rest of the pipeline.
SAMPLE_SIZE_ALIASES = ["Sample_Size_Total", "SampleSizeTotal"]
SEGMENT_ID_ALIASES = ["DISSOLVE_ID", "OBJECTID"]
ROAD_CLASS_ALIASES = ["RoadClass", "class"]
ROAD_NAME_ALIASES = ["names_primary", "english_ro"]


def load_geojson(path):
    """Read a GeoJSON file into a GeoDataFrame, reprojecting to EPSG:4326 if needed."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def detect_country(gdf):
    """Infer India vs Thailand from the geometry bounds of a GeoDataFrame."""
    minx, miny, maxx, maxy = gdf.total_bounds
    for country, bounds in COUNTRY_BOUNDS.items():
        lon_lo, lon_hi = bounds["lon"]
        lat_lo, lat_hi = bounds["lat"]
        if lon_lo <= minx and maxx <= lon_hi and lat_lo <= miny and maxy <= lat_hi:
            return country
    # Fall back to whichever bounding box the centroid of the data falls into.
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    for country, bounds in COUNTRY_BOUNDS.items():
        lon_lo, lon_hi = bounds["lon"]
        lat_lo, lat_hi = bounds["lat"]
        if lon_lo <= cx <= lon_hi and lat_lo <= cy <= lat_hi:
            return country
    raise ValueError("Could not determine country from coordinate bounds")


def _first_present(gdf, aliases):
    """Return the first column name from aliases that exists in gdf, else None."""
    for name in aliases:
        if name in gdf.columns:
            return name
    return None


def harmonize_schema(gdf, country):
    """Add unified segment_id, road_class, Sample_Size_Total, and country columns."""
    gdf = gdf.copy()
    gdf["country"] = country

    id_col = _first_present(gdf, SEGMENT_ID_ALIASES)
    gdf["segment_id"] = gdf[id_col].astype(str)

    class_col = _first_present(gdf, ROAD_CLASS_ALIASES)
    gdf["road_class"] = gdf[class_col]

    # India carries a name in names_primary; Thailand's equivalent is english_ro
    # (a truncated "english_road" field from the Agilysis export). Neither is
    # populated for every segment, so unnamed segments fall back to a
    # class + id label rather than a blank/NaN in the UI.
    name_col = _first_present(gdf, ROAD_NAME_ALIASES)
    road_name = gdf[name_col] if name_col else pd.Series(pd.NA, index=gdf.index)
    has_name = road_name.notna() & (road_name.astype(str).str.strip() != "")
    fallback = gdf["road_class"].fillna("Road") + " segment " + gdf["segment_id"]
    gdf["road_name"] = road_name.where(has_name, fallback)
    # Flags which segments got the generated label rather than a real source
    # name, so a later step (see src/road_names.py) knows which ones are
    # worth trying to backfill from an external name source.
    gdf["road_name_is_fallback"] = ~has_name

    size_col = _first_present(gdf, SAMPLE_SIZE_ALIASES)
    gdf["Sample_Size_Total"] = pd.to_numeric(gdf[size_col], errors="coerce")

    # SpeedLimit is stored as text in the Maharashtra file; coerce both to numeric.
    gdf["SpeedLimit"] = pd.to_numeric(gdf["SpeedLimit"], errors="coerce")
    gdf["F85thPercentileSpeed"] = pd.to_numeric(gdf["F85thPercentileSpeed"], errors="coerce")

    # The data guide says to ignore RoadLength (km) in favour of Shape_Length
    # (metres). They encode the same value here, but Shape_Length is the
    # authoritative field, so RoadLength_km is derived from it.
    gdf["RoadLength_km"] = gdf["Shape_Length"] / 1000.0

    # UrbanPC only exists for India; Thailand instead has a categorical LandUse field.
    if "UrbanPC" not in gdf.columns:
        gdf["UrbanPC"] = (gdf["LandUse"] == "URBAN").astype(float)

    # RankedPercentile is a 0-1 fraction in the India file but 0-100 in the
    # Thailand file. Harmonise to 0-100 so cross-country correlation is valid.
    max_val = gdf["RankedPercentile"].max(skipna=True)
    if pd.notna(max_val) and max_val <= 1.0:
        gdf["RankedPercentile"] = gdf["RankedPercentile"] * 100

    return gdf


def filter_reliable(gdf):
    """Split a harmonised GeoDataFrame into (reliable, low_confidence) segments."""
    mask = pd.Series(True, index=gdf.index)
    if "ExcludeFromSpeedSPI" in gdf.columns:
        mask &= gdf["ExcludeFromSpeedSPI"].fillna(1) == 0
    if "AnalysisStatus" in gdf.columns:
        mask &= gdf["AnalysisStatus"] == "Valid"
    mask &= gdf["Sample_Size_Total"] >= 1000

    reliable = gdf[mask].copy()
    low_confidence = gdf[~mask].copy()
    low_confidence["low_confidence"] = True
    return reliable, low_confidence


def load_helmet_layer(gpkg_path, layer, country):
    """Load a province/zone helmet-wearing layer and return it with a 0-1 helmet_rate column."""
    zones = gpd.read_file(gpkg_path, layer=layer)
    if country == "India":
        zones = zones.rename(columns={"AllRidersSPI": "helmet_rate"})
    else:
        zones = zones[zones["INCLUDE"] == "Y"].copy()
        zones["helmet_rate"] = zones["overall_helmet_use_pct"] / 100.0
    return zones.to_crs("EPSG:4326")[["helmet_rate", "geometry"]]


def assign_zone_attribute(segments, zones, metric_crs):
    """Spatially join each segment's representative point to a zone polygon (nearest as fallback)."""
    points = segments.copy()
    points["geometry"] = points.geometry.representative_point()

    joined = gpd.sjoin(points, zones, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.drop(columns=["index_right"], errors="ignore")

    missing = joined["helmet_rate"].isna()
    if missing.any():
        # Segments whose centroid misses every polygon (slivers/coastline) get
        # the nearest zone instead of being left null.
        nearest = gpd.sjoin_nearest(
            points.loc[missing].to_crs(metric_crs),
            zones.to_crs(metric_crs),
            how="left",
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[missing, "helmet_rate"] = nearest["helmet_rate"].reindex(joined.loc[missing].index).values

    return joined["helmet_rate"].reindex(segments.index)
