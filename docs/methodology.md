# Methodology

## Overview

This project scores road segments in India (Maharashtra) and Thailand on how
misaligned their posted speed limits are with Safe System principles and
vulnerable road user (VRU) exposure. It does **not** train a predictive
model — the Speed Safety Score is a transparent, hand-weighted formula built
from four risk components, chosen so the reasoning behind every segment's
score is auditable. See `src/train_model.py` for an intentionally unfinished
scaffold if a future iteration wants to fit an actual predictive model.

## Data sources and schema reconciliation

The two source files were exported at different times by different teams and
**do not share a schema**. Thailand's columns match the Agilysis data-guide
PDF (`RoadClass`, `SampleSizeTotal`, no `DISSOLVE_ID`/`UrbanPC`); Maharashtra's
match the field list in the challenge brief (`DISSOLVE_ID`, `class`,
`UrbanPC`, `ExcludeFromSpeedSPI`, `Sample_Size_Total`). `src/utils.py:harmonize_schema`
reconciles both onto a common column set:

| Unified column | India (Maharashtra) source | Thailand source |
|---|---|---|
| `segment_id` | `DISSOLVE_ID` | `OBJECTID` |
| `road_class` | `class` (== `RoadClass`, verified identical) | `RoadClass` |
| `Sample_Size_Total` | `Sample_Size_Total` | `SampleSizeTotal` |
| `RoadLength_km` | `Shape_Length / 1000` | `Shape_Length / 1000` |
| `UrbanPC` | native field | derived: 1.0 if `LandUse == 'URBAN'` else 0.0 |

The two files are **disjoint geographic datasets** (different countries), so
the "join match rate" check the brief asks for returns ~0% overlap by
design — they are concatenated, not joined on a shared key.

`RoadLength` is explicitly marked "ignore, use Shape Length" in the Agilysis
data guide. Both fields were verified to encode the same length (Shape_Length
in metres, RoadLength in km, related by an exact ×1000 factor), so
`RoadLength_km` is derived from `Shape_Length` throughout.

`RankedPercentile` is a 0–1 fraction in the Maharashtra file but a 0–100 scale
in the Thailand file; it is rescaled to 0–100 for both before any
cross-country comparison.

## Known data anomaly

Some segments report `F85thPercentileSpeed` more than 20 km/h above
`SpeedLimit` while `PercentOverLimit` reads 0. These are flagged in
notebook 01 but **not excluded** — the underlying cause (sparse sampling,
sample taken at a different point than the posted limit applies, etc.) isn't
resolvable from the data alone.

## Reliability filter

A segment is "reliable" if (where the column exists) `ExcludeFromSpeedSPI ==
0`, `AnalysisStatus == 'Valid'`, and `Sample_Size_Total >= 1000`. Everything
else is "low confidence" and excluded from scoring, but kept in
`data/processed/segments_low_confidence.geojson` and shown on the map as a
separate, off-by-default "Insufficient data" layer.

## Feature definitions

- **`speed_gap`** = `max(F85thPercentileSpeed - SpeedLimit, 0)`.
  `speed_gap_norm` = `min(speed_gap / 20, 1)` — an absolute anchor where
  driving 20+ km/h over the limit is a full-strength signal. Only positive
  gaps are a risk signal — a segment where traffic travels *below* the
  limit gets 0, not a negative score.
- **`road_mismatch`** compares the posted `SpeedLimit` against the Safe
  System speed range for its road class. It is only positive when the
  posted limit **exceeds** the class ceiling (motorway 110, trunk 90,
  primary 70, secondary 60, tertiary 50, residential 30, living_street 20,
  unclassified 40): `min(km/h over the ceiling / 30, 1)`, so a limit
  posted 30+ km/h above its class ceiling is a full-strength signal.
  **Note:** the brief's worked example
  claims a secondary road posted at 55 km/h is "5 km/h above" the 60 km/h
  secondary ceiling — that's arithmetically backwards (55 < 60). Per the
  stated definition, that case correctly scores `road_mismatch = 0`; this
  was verified against the 1,148 real Maharashtra segments posted at 55
  km/h on secondary roads.
- **`urban_flag`** = 1.0 if `UrbanPC > 0.5` else 0.0.
- **`vru_exposure`** blends `urban_flag` (40% weight) with a region-level
  low-helmet-compliance risk signal (60% weight): `1 - helmet_wearing_rate`,
  spatially joined from the helmet-wearing survey layers in
  `Archive/*.gpkg` and min-max normalised. **Resolution differs sharply
  between countries** — 4 zones for Maharashtra (Mumbai, Pune, Maharashtra
  Rural, Maharashtra Urban) vs. 77 provinces for Thailand — so this is a
  coarse proxy, especially for India, not a precise exposure model. If no
  helmet layer is supplied, the function falls back to `urban_flag` alone
  (see `compute_vru_exposure` in `src/features.py`).
- **`bio_risk`** = a fatality-probability lookup on the exposure speed
  `max(F85thPercentileSpeed, SpeedLimit)` (≤20: 0.05, ≤30: 0.10, ≤40: 0.30,
  ≤50: 0.80, ≤60: 0.90, >60: 1.00), multiplied by `vru_exposure`. Using the
  higher of measured operating speed and posted limit captures both roads
  where traffic already runs at lethal speeds and roads whose limit permits
  them.
- **`confidence_weight`** = 0.5 if `RoadLength_km > 10` else 1.0 — a
  data-confidence *flag* for long segments where a sparse point sample may
  not represent the whole segment. It is reported alongside the score but
  no longer multiplies it (see "Score recalibration" below).
- **`mapillary_url`** is built from the centroid of the two endpoint
  coordinates in `StreetImageLink` (the data guide describes this field as
  endpoint lon/lat pairs, not a sorted bounding box — but centroid averaging
  produces the correct midpoint either way).
- **`recommended_speed_limit`** interpolates within the segment's road-class
  Safe System range (the same `motorway`/`trunk`/.../`unclassified` ranges
  used by `road_mismatch`), pulled toward the class **minimum** as
  `vru_exposure` rises toward 1 and toward the class **maximum** as it falls
  toward 0:
  `class_min + (class_max - class_min) * (1 - vru_exposure)`, rounded to the
  nearest 10 km/h (how limits are actually posted). Segments whose
  `road_class` has no defined Safe System range get `NaN`.
- **`speed_limit_gap`** = posted `SpeedLimit` minus `recommended_speed_limit`.
  Positive means the posted limit sits above what Safe System principles
  suggest for that segment's vulnerable-road-user exposure — this is the
  field to hand a transport ministry official for "what should this
  segment's limit actually be." Across the 14,546 reliable segments: mean
  gap +12.8 km/h, 73.3% of segments posted above their recommendation.

## Speed Safety Score

```
speed_safety_score = round(
    (0.30 * speed_gap_norm
   + 0.30 * road_mismatch
   + 0.40 * bio_risk)
  * 100,
  1
)
```

Risk tiers: High risk ≥ 70, Medium risk ≥ 40, Low risk < 40, Insufficient
data for segments excluded by the reliability filter.

Design decisions behind this formula (revised after a first iteration —
see "Score recalibration" below):

- **`vru_exposure` is not a separate term.** It already scales `bio_risk`
  (consequence × exposure); carrying it twice double-counted the signal
  and compressed the score's usable range. `bio_risk` gets the largest
  weight because "lethal speeds where unprotected people are" is the core
  Safe System concern; the two limit-misalignment signals split the rest.
- **`confidence_weight` does not multiply the score.** Discounting risk for
  data uncertainty made genuinely dangerous long segments invisible. It is
  reported alongside the score as a data-confidence flag instead.
- **All components sit on absolute anchors** (20 km/h over-limit gap and
  30 km/h over-ceiling posting saturate their signals), so a given score
  means the same thing in any country the pipeline is applied to —
  a requirement for the challenge's scalability goal.

## Score recalibration (why the formula changed)

The first iteration produced **zero** High-risk and only 382 Medium-risk
segments out of 14,546, with a maximum observed score of 53.3 — the 40/70
tier thresholds were structurally unreachable. Component-level analysis
found four compounding causes:

1. `speed_gap_norm` was min-max scaled against the dataset maximum (an
   88 km/h outlier), so a serious 20 km/h over-limit gap scored only 0.23
   — and the score's meaning depended on the dataset it was computed in.
2. `road_mismatch` divided the over-posting by the class ceiling, capping
   the achievable value at ~0.5 for every real over-posting — silently
   halving that component's weight.
3. `vru_exposure` was double-counted (inside `bio_risk` and as its own
   term) yet rarely approached 1.0, diluting both terms.
4. `confidence_weight` halved the score of 2,778 segments (19%) for being
   long — conflating data uncertainty with safety.

The recalibrated formula fixes all four (absolute anchors, fixed 30 km/h
mismatch denominator, exposure folded into `bio_risk` only, confidence as
a flag). `bio_risk` also now evaluates the fatality probability at
`max(F85, SpeedLimit)` — the speed VRUs are actually exposed to — rather
than at the posted limit alone. The highest-scoring segments under the
new formula are urban secondary/primary roads posted 30 km/h above their
Safe System ceiling with 85th-percentile speeds of 104–110 km/h and high
VRU exposure — exactly the profile the challenge asks to surface.

## Validation results (current run)

- **Correlation with `RankedPercentile`: -0.01** (effectively
  uncorrelated). `RankedPercentile` ranks segments by **travel volume
  share** (per the Agilysis data guide: "allows presentation of roads by
  percentage traffic"), not by safety risk. A busy, well-engineered
  motorway can carry a large share of national travel while being
  comparatively safe, while a low-traffic rural secondary road posted above
  its Safe System range scores high on risk but negligible on travel share.
  The two metrics measure different things, so near-zero correlation is
  the expected outcome, and it is reported here rather than masked.
- **Risk tier distribution**: 79 High risk (0.5%), 5,476 Medium risk
  (37.6%), 8,991 Low risk among the 14,546 reliable segments (max score
  observed: 77.9). The small High tier is a feature, not a bug: it is a
  reviewable priority list for a ministry, not a statistical artifact —
  every member is posted ≥30 km/h above its Safe System class ceiling with
  measured speeds far above even that, in high-VRU-exposure areas.
- **Sensitivity analysis**: 18 valid weight combinations (±0.10 per weight,
  0.05 steps, summing to 1.0) were tested against the baseline top-20%
  highest-scoring segments. Average overlap: **79.7%** — above the 70%
  threshold, so the score is robust to reasonable weight re-calibration.

## Map performance trade-off

Embedding all ~70,000 segments (reliable + low-confidence) in one Folium
HTML file at full geometry/attribute fidelity produced a 61MB file —
impractical for GitHub Pages and slow to pan/zoom in a browser. The
low-confidence ("Insufficient data") layer is therefore: (a) simplified more
aggressively (0.0015° tolerance vs. 0.0003° for scored segments), (b) given a
minimal tooltip with no popup, and (c) off by default via the layer control.
This brought the file to ~30MB while keeping every segment inspectable on
demand.

## Limitations

- `SpeedLimit` and the original `LandUse` classification are estimates from
  secondary sources (per the Agilysis data guide) and should be treated as
  approximate; `UrbanPC` is the more reliable numeric field, used here.
  These are limitations of the upstream data, not of this pipeline.
- The VRU exposure proxy relies on regional helmet-wearing surveys, not
  segment-level pedestrian/cyclist counts — a genuine exposure model (e.g.
  population density, footfall, school/market proximity) is future work.
- This is a rule-based score, not a validated predictive model. See
  `src/train_model.py` for a deliberately unfinished scaffold to take this
  further with real outcome data.
