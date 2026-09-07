"""
test_geo.py — unit tests for src/geo.py (area -> map coordinate).

No network. Just checks the lookup, the fallback, the jitter's determinism,
and that an explicit lat/lng on a record wins.

Run:  pytest tests/test_geo.py -v
"""

import geo


def test_known_area_lands_near_its_centroid():
    """A recognised town resolves within the jitter radius of its centroid."""
    base = geo.AREA_COORDS["toa payoh"]
    loc = geo.locate({"name": "Some Day Centre (Toa Payoh)", "area": "Toa Payoh"})
    assert abs(loc["lat"] - base[0]) <= geo._JITTER_DEG + 1e-9
    assert abs(loc["lng"] - base[1]) <= geo._JITTER_DEG + 1e-9


def test_islandwide_area_falls_back_near_singapore_centre():
    """'Islandwide (care worker travels to the home)' has no matching key -> SG centre (+ jitter)."""
    loc = geo.locate({"name": "Homage Home-Based Respite Care",
                      "area": "Islandwide (care worker travels to the home)"})
    assert abs(loc["lat"] - geo.SINGAPORE_CENTRE[0]) <= geo._JITTER_DEG + 1e-9
    assert abs(loc["lng"] - geo.SINGAPORE_CENTRE[1]) <= geo._JITTER_DEG + 1e-9


def test_jurong_east_prefers_the_jurong_east_centroid():
    """Substring match picks the most specific key that is contained in the area."""
    loc = geo.locate({"name": "All Saints Home (Jurong East)", "area": "Jurong East"})
    base = geo.AREA_COORDS["jurong east"]
    assert abs(loc["lat"] - base[0]) <= geo._JITTER_DEG + 1e-9


def test_jitter_is_deterministic_and_separates_same_town_services():
    """Two services in one town get stable but distinct coordinates."""
    a1 = geo.locate({"name": "Centre A", "area": "Bedok"})
    a2 = geo.locate({"name": "Centre A", "area": "Bedok"})
    b = geo.locate({"name": "Centre B", "area": "Bedok"})
    assert a1 == a2                       # same input -> same output across calls
    assert (a1["lat"], a1["lng"]) != (b["lat"], b["lng"])  # different names -> different pins


def test_explicit_coordinates_on_the_record_win():
    """If a record ever carries real lat/lng, geo.locate uses them verbatim."""
    loc = geo.locate({"name": "X", "area": "Bedok", "lat": 1.23456, "lng": 103.65432})
    assert loc == {"lat": 1.23456, "lng": 103.65432}


def test_unknown_area_falls_back_to_singapore_centre():
    loc = geo.locate({"name": "Mystery", "area": "Atlantis"})
    assert abs(loc["lat"] - geo.SINGAPORE_CENTRE[0]) <= geo._JITTER_DEG + 1e-9
    assert abs(loc["lng"] - geo.SINGAPORE_CENTRE[1]) <= geo._JITTER_DEG + 1e-9
