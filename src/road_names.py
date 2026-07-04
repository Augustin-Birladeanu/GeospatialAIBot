"""Backfill sparse road names using Overture Maps' own road-name data,
resolved to English wherever possible.

The Speed Safety Score's road_name field (harmonize_schema) only carries
whatever the challenge's raw exports populated -- names_primary for India,
english_ro for Thailand -- and both are sparsely filled: ~29% of reliable
India segments and ~31% of reliable Thailand segments have a real name, the
rest get a generated "{road_class} segment {segment_id}" label. A live check
against Overture Maps' current road data for a Maharashtra sample found only
~34% name coverage there too, confirming this is a genuine OpenStreetMap/
Overture road-naming completeness gap for these regions' minor and rural
roads -- not an artifact of this specific export -- so this backfill is a
real but modest improvement, not a full fix. Most unnamed segments are
unnamed everywhere because the road genuinely has no name in the real
world; no data source changes that.

English resolution: Overture's `primary` name is the *local* name, which for
India is already in Latin script 98.7% of the time but for Thailand is Thai
script 86.6% of the time. This module prefers, in order: (1) `primary` if
it's already Latin-script, (2) Overture's own tagged English `common` name
if present (covers ~60% of Thai segments that need it), (3) a mechanical
script-to-Latin romanization (pythainlp's Royal Institute romanizer for
Thai, indic_transliteration/ITRANS for Devanagari) as a last resort. Step 3
is a phonetic approximation, not an authoritative English name (e.g.
Devanagari's implicit final vowel makes "road" transliterate as "Roda") --
labelled as such, not passed off as official.

Fetching Overture's road segments for a whole state/country isn't cheap
enough to do inline (a naive full download is tens of GB), so
scripts/fetch_overture_names.py fetches only what this needs -- named,
subtype=road, our used road classes -- server-side filtered, and this module
does the local English-resolution and spatial match against that file.
"""

import re

import geopandas as gpd
import pandas as pd
from shapely import wkb

# Road centerlines from two different sources (this project's segments vs.
# Overture's) rarely align exactly, so the match uses a small tolerance
# rather than requiring the geometries to touch.
MAX_MATCH_DISTANCE_M = 50

_LATIN_RE = re.compile(r"[A-Za-z0-9 .,'\-()/&]+")
_NON_LATIN_CHAR_RE = re.compile(r"[^A-Za-z0-9 .,'\-()/&–—]")


def _is_latin(text):
    return bool(text) and bool(_LATIN_RE.fullmatch(text))


def _is_mostly_latin(text, max_non_latin_fraction=0.03):
    """True if a romanization attempt is clean enough to show (allows an en/em dash,
    but rejects results with leftover script characters or garbled tokenizer artifacts)."""
    if not text:
        return False
    non_latin_chars = len(_NON_LATIN_CHAR_RE.findall(text))
    return (non_latin_chars / len(text)) <= max_non_latin_fraction


def _find_common_language(names, language):
    """Look up a language-tagged variant in Overture's names.common list of [lang, value] pairs."""
    common = names.get("common") if isinstance(names, dict) else None
    if not isinstance(common, list):
        return None
    for pair in common:
        try:
            lang, value = pair
        except (TypeError, ValueError):
            continue
        if lang == language:
            return value
    return None


def _romanize_thai(text):
    from pythainlp.tokenize import word_tokenize
    from pythainlp.transliterate.royin import romanize as royin_romanize

    words = []
    for word in word_tokenize(text, engine="newmm"):
        word = word.strip()
        if not word:
            continue
        words.append(word if word.isascii() else royin_romanize(word).capitalize())
    return " ".join(words) or None


def _romanize_devanagari(text):
    from indic_transliteration import sanscript

    itrans = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    return re.sub(r"[a-z]+", lambda m: m.group(0).capitalize(), itrans) or None


_ROMANIZERS = {"India": _romanize_devanagari, "Thailand": _romanize_thai}


def resolve_english_text(text, country):
    """Return text unchanged if already Latin-script, else a best-effort romanization, else None.

    Used both for Overture's names.primary (see _resolve_english_name) and
    for the raw challenge exports' own name fields (names_primary /
    english_ro), which are overwhelmingly but not entirely already in
    English (94.6% of India's, 99.8% of Thailand's) -- the same
    romanize-or-reject logic applies to that small non-English remainder.
    """
    if _is_latin(text):
        return text
    romanize = _ROMANIZERS.get(country)
    if not romanize or not text:
        return None
    try:
        romanized = romanize(text)
    except Exception:
        return None
    # Reject anything that's still substantially non-Latin (mixed-script
    # border-area names, tokenizer glitches) rather than show garbled text --
    # those segments keep the generated fallback label instead.
    return romanized if _is_mostly_latin(romanized) else None


def _resolve_english_name(names, country):
    """Best-effort English name for one Overture names struct: Latin primary, else tagged English, else romanized."""
    if not isinstance(names, dict):
        return None
    primary = names.get("primary")
    if _is_latin(primary):
        return primary
    en_common = _find_common_language(names, "en")
    if en_common:
        return en_common
    return resolve_english_text(primary, country) if primary else None


def load_overture_names(parquet_path, country):
    """Load a fetch_overture_names.py output into an English-resolved line GeoDataFrame."""
    df = pd.read_parquet(parquet_path)
    df["geometry"] = df["geometry"].map(wkb.loads)

    # Resolve on the unique set of raw name structs first -- many segments
    # share the same road name, and romanization is the expensive step.
    primary_key = df["names"].map(lambda n: n.get("primary") if isinstance(n, dict) else None)
    first_seen = df.groupby(primary_key, dropna=False)["names"].first()
    resolved_by_primary = first_seen.map(lambda n: _resolve_english_name(n, country))
    df["name"] = primary_key.map(resolved_by_primary)

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
