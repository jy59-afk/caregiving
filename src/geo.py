"""
geo.py — turns a service record's `area` (a Singapore town / planning-area
name) into an approximate map coordinate, so the web frontend can pin the
shortlist on a Leaflet map.

Why this is a lookup and not real geocoding: `data/services.json` stores a
town name, not a street address the vector index needs (a few records carry a
full address inside `description`, most do not). Town-centroid coordinates are
accurate enough to show a caregiver *roughly where* each option sits and how
the four compare — which is all the map is for. When a record ever gains a
real `lat`/`lng`, `locate()` prefers those.

Nothing here touches a network or a model.
"""

from __future__ import annotations

import hashlib  # deterministic per-service jitter so co-located pins don't overlap exactly

# --------------------------------------------------------------------------
# Approximate centroids (WGS84 lat, lng) for every `area` value that appears
# in data/services.json, plus a handful of near-synonyms. Hand-placed from
# public map references; good to ~1 km, which is the resolution the map needs.
# --------------------------------------------------------------------------
AREA_COORDS: dict[str, tuple[float, float]] = {
    "toa payoh": (1.3343, 103.8563),
    "bishan": (1.3526, 103.8352),
    "balestier": (1.3260, 103.8480),
    "novena": (1.3203, 103.8438),
    "potong pasir": (1.3312, 103.8690),
    "serangoon": (1.3554, 103.8679),
    "ang mo kio": (1.3691, 103.8454),
    "hougang": (1.3712, 103.8925),
    "yishun": (1.4304, 103.8354),
    "sembawang": (1.4491, 103.8200),
    "woodlands": (1.4382, 103.7890),
    "bukit batok": (1.3590, 103.7637),
    "bukit panjang": (1.3774, 103.7719),
    "choa chu kang": (1.3840, 103.7470),
    "clementi": (1.3151, 103.7654),
    "jurong": (1.3329, 103.7436),
    "jurong east": (1.3329, 103.7436),
    "jurong west": (1.3404, 103.7090),
    "bukit timah": (1.3294, 103.8021),
    "queenstown": (1.2942, 103.8059),
    "bukit merah": (1.2819, 103.8239),
    "outram": (1.2793, 103.8395),
    "geylang": (1.3183, 103.8870),
    "joo chiat": (1.3117, 103.9021),
    "bedok": (1.3236, 103.9273),
    "simei": (1.3434, 103.9532),
    "tampines": (1.3496, 103.9568),
    "pasir ris": (1.3721, 103.9474),
    "changi": (1.3500, 103.9880),
}

# Fallback when the area is unknown or islandwide (e.g. Homage sends a worker
# to the home): the geographic centre of Singapore's residential belt.
SINGAPORE_CENTRE: tuple[float, float] = (1.3521, 103.8198)

# How far (in degrees, ~111 km per degree of latitude) a pin may be nudged
# from its town centroid. ~0.006 deg ≈ 650 m — enough to fan out the several
# nursing homes that share one town, small enough to stay in that town.
_JITTER_DEG = 0.006


def _centroid_for_area(area: str | None) -> tuple[float, float]:
    """
    Resolve a free-text area to a centroid: exact key first, then a substring
    match (handles "Islandwide (care worker travels to the home)" and
    "Jurong East" vs "Jurong"), then the Singapore-centre fallback.
    """
    if not area:  # None or "" — nothing to match on
        return SINGAPORE_CENTRE
    key = area.strip().lower()  # normalise for the dict keys, which are all lowercase
    if key in AREA_COORDS:  # the common case — the area is a known town
        return AREA_COORDS[key]
    for name, coord in AREA_COORDS.items():  # e.g. "jurong east ..." contains "jurong east"
        if name in key:  # substring hit is good enough for the messy values
            return coord
    return SINGAPORE_CENTRE  # unknown town / "islandwide" — drop it in the middle


def _jitter(name: str) -> tuple[float, float]:
    """
    A small, deterministic (lat, lng) offset derived from the service name, so
    two services in the same town render as two distinct pins but never move
    between page loads.
    """
    digest = hashlib.sha1(name.encode("utf-8")).digest()  # stable hash of the name
    # Map two bytes to [-1, 1) each, then scale to the jitter radius.
    lat_frac = digest[0] / 128.0 - 1.0  # byte 0 -> latitude nudge
    lng_frac = digest[1] / 128.0 - 1.0  # byte 1 -> longitude nudge
    return lat_frac * _JITTER_DEG, lng_frac * _JITTER_DEG


def locate(service: dict) -> dict[str, float]:
    """
    Return `{"lat": ..., "lng": ...}` for one service record.

    Uses explicit `lat`/`lng` on the record if present; otherwise the town
    centroid for `area`, plus a deterministic per-name jitter so co-located
    services don't stack into a single pin.
    """
    if service.get("lat") is not None and service.get("lng") is not None:  # a real coordinate wins
        return {"lat": float(service["lat"]), "lng": float(service["lng"])}
    base_lat, base_lng = _centroid_for_area(service.get("area"))  # town-level position
    d_lat, d_lng = _jitter(service.get("name", service.get("area", "")))  # spread same-town pins apart
    return {"lat": round(base_lat + d_lat, 6), "lng": round(base_lng + d_lng, 6)}
