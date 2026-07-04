"""Download named Overture Maps road segments for a bbox, filtered server-side
to just what a road-name backfill needs (named, subtype=road, our road classes).

A full unfiltered bbox pull of Overture's transportation segments is tens of
GB per country/state, so this pushes the subtype/class/"has a name" filters
and a narrow column projection down to the scan itself rather than
downloading everything and filtering locally (see src/road_names.py for the
matching step that consumes this output).

Usage: python scripts/fetch_overture_names.py "<xmin,ymin,xmax,ymax>" <out.parquet>
Used to produce data/raw/overture_road_names_india.parquet and
data/raw/overture_road_names_thailand.parquet -- re-run only if those need
refreshing from a newer Overture release.
"""
import sys
import time

import pyarrow.compute as pc
import pyarrow.parquet as pq
from overturemaps.core import _prepare_query

ROAD_CLASSES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "residential", "living_street", "unclassified",
]


def main(bbox, out_path):
    t0 = time.time()
    dataset, bbox_filter = _prepare_query("segment", bbox=bbox)
    combined = (
        bbox_filter
        & (pc.field("subtype") == "road")
        & pc.field("class").isin(ROAD_CLASSES)
        & pc.field("names").is_valid()
    )
    scanner = dataset.scanner(
        columns=["id", "names", "class", "geometry"], filter=combined, use_threads=True
    )
    with pq.ParquetWriter(out_path, scanner.projected_schema) as writer:
        n = 0
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            writer.write_batch(batch)
            n += batch.num_rows
            print(f"  ...{n} rows so far ({round(time.time()-t0)}s)", flush=True)
    print(f"DONE {out_path}: {n} rows in {round(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    bbox = tuple(float(x) for x in sys.argv[1].split(","))
    main(bbox, sys.argv[2])
