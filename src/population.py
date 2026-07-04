"""Population-raster enrichment for road segments."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats


def add_population_data(gdf, raster_paths):
    """Add raster population statistics and a 0-1 exposure value per segment.

    ``raster_paths`` maps each value in the ``country`` column to its
    population GeoTIFF. Raster values are treated as population counts per
    cell. Each segment receives the mean and maximum of all cells it touches.
    Exposure is log-scaled and normalized independently within each country
    against its 95th percentile, limiting the influence of extreme cells.
    """
    required = {"country", "geometry"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = gdf.copy()
    out["population_mean"] = np.nan
    out["population_max"] = np.nan

    for country, raster_path in raster_paths.items():
        mask = out["country"] == country
        if not mask.any():
            continue

        path = Path(raster_path)
        if not path.is_file():
            raise FileNotFoundError(f"Population raster for {country} not found: {path}")

        stats = zonal_stats(
            out.loc[mask, "geometry"],
            path,
            stats=["mean", "max"],
            all_touched=True,
            geojson_out=False,
        )
        out.loc[mask, "population_mean"] = [
            stat["mean"] if stat["mean"] is not None and stat["mean"] >= 0 else np.nan
            for stat in stats
        ]
        out.loc[mask, "population_max"] = [
            stat["max"] if stat["max"] is not None and stat["max"] >= 0 else np.nan
            for stat in stats
        ]

    out["population_exposure"] = 0.0
    for country, index in out.groupby("country").groups.items():
        values = pd.to_numeric(out.loc[index, "population_mean"], errors="coerce").clip(lower=0)
        logged = np.log1p(values)
        ceiling = logged.quantile(0.95)
        if pd.notna(ceiling) and ceiling > 0:
            out.loc[index, "population_exposure"] = (logged / ceiling).clip(0, 1)

    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)
