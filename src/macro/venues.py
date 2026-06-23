"""AFL venue -> city/state geography and a travel-burden model.

This module is the single source of truth for *where* AFL games are played and
how far each team has to travel to get there. It is consumed by
:mod:`src.macro.elo` to turn a flat home-ground advantage (HGA) into a
venue/travel-aware one.

Design notes
------------
* Coordinates are city-level (good enough for travel burden; the error from
  using a club's training base vs. the actual stadium is tiny next to
  inter-city distances).
* ``travel_km`` measures great-circle distance from a team's home base to the
  match venue. The away team almost always travels further; when a "home" game
  is staged out of the home team's state (Gather Round, Tasmania, China, ...)
  the home team travels too, which correctly shrinks its advantage.
* Everything here is pure/stateless so it is trivially testable and has no DB
  dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class City:
    name: str
    state: str  # AFL state code: VIC, SA, WA, QLD, NSW, ACT, TAS, NT, OS (overseas)
    lat: float
    lon: float


# Canonical city coordinates (decimal degrees).
CITIES: dict[str, City] = {
    "Melbourne": City("Melbourne", "VIC", -37.8136, 144.9631),
    "Geelong": City("Geelong", "VIC", -38.1580, 144.3540),
    "Ballarat": City("Ballarat", "VIC", -37.5622, 143.8503),
    "Adelaide": City("Adelaide", "SA", -34.9285, 138.6007),
    "Mount Barker": City("Mount Barker", "SA", -35.0667, 138.8597),
    "Barossa": City("Barossa", "SA", -34.5340, 138.9520),
    "Perth": City("Perth", "WA", -31.9505, 115.8605),
    "Bunbury": City("Bunbury", "WA", -33.3271, 115.6414),
    "Brisbane": City("Brisbane", "QLD", -27.4858, 153.0381),
    "Gold Coast": City("Gold Coast", "QLD", -28.0067, 153.3686),
    "Cairns": City("Cairns", "QLD", -16.9203, 145.7710),
    "Townsville": City("Townsville", "QLD", -19.2590, 146.8169),
    "Sydney": City("Sydney", "NSW", -33.8688, 151.2093),
    "Canberra": City("Canberra", "ACT", -35.2809, 149.1300),
    "Hobart": City("Hobart", "TAS", -42.8821, 147.3272),
    "Launceston": City("Launceston", "TAS", -41.4332, 147.1441),
    "Darwin": City("Darwin", "NT", -12.4634, 130.8456),
    "Alice Springs": City("Alice Springs", "NT", -23.6980, 133.8807),
    "Shanghai": City("Shanghai", "OS", 31.2304, 121.4737),
}


# Venue (exactly as stored in ``Match.venue``) -> city key.
VENUE_CITY: dict[str, str] = {
    "M.C.G.": "Melbourne",
    "Docklands": "Melbourne",
    "Marvel Stadium": "Melbourne",
    "Kardinia Park": "Geelong",
    "GMHBA Stadium": "Geelong",
    "Eureka Stadium": "Ballarat",
    "Mars Stadium": "Ballarat",
    "Adelaide Oval": "Adelaide",
    "Norwood Oval": "Adelaide",
    "Adelaide Hills": "Mount Barker",
    "Barossa Park": "Barossa",
    "Optus Stadium": "Perth",
    "Perth Stadium": "Perth",
    "Hands Oval": "Bunbury",
    "Gabba": "Brisbane",
    "Carrara": "Gold Coast",
    "Cazaly's Stadium": "Cairns",
    "Riverway Stadium": "Townsville",
    "S.C.G.": "Sydney",
    "Stadium Australia": "Sydney",
    "Sydney Showground": "Sydney",
    "Manuka Oval": "Canberra",
    "UNSW Canberra Oval": "Canberra",
    "Bellerive Oval": "Hobart",
    "York Park": "Launceston",
    "University of Tasmania Stadium": "Launceston",
    "Marrara Oval": "Darwin",
    "Traeger Park": "Alice Springs",
    "Jiangwan Stadium": "Shanghai",
    "Adelaide Arena at Jiangwan Stadium": "Shanghai",
}


# Team -> home base city key.
TEAM_CITY: dict[str, str] = {
    "Adelaide": "Adelaide",
    "Port Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane",
    "Gold Coast": "Gold Coast",
    "Carlton": "Melbourne",
    "Collingwood": "Melbourne",
    "Essendon": "Melbourne",
    "Hawthorn": "Melbourne",
    "Melbourne": "Melbourne",
    "North Melbourne": "Melbourne",
    "Richmond": "Melbourne",
    "St Kilda": "Melbourne",
    "Western Bulldogs": "Melbourne",
    "Geelong": "Geelong",
    "Fremantle": "Perth",
    "West Coast": "Perth",
    "Greater Western Sydney": "Sydney",
    "Sydney": "Sydney",
}


_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def venue_city(venue: str | None) -> City | None:
    """Resolve a venue string to its :class:`City`, or ``None`` if unknown."""
    if not venue:
        return None
    key = VENUE_CITY.get(venue) or VENUE_CITY.get(venue.strip())
    return CITIES.get(key) if key else None


def team_city(team: str) -> City | None:
    """Resolve a team to its home-base :class:`City`."""
    key = TEAM_CITY.get(team)
    return CITIES.get(key) if key else None


def venue_state(venue: str | None) -> str | None:
    city = venue_city(venue)
    return city.state if city else None


def travel_km(team: str, venue: str | None) -> float:
    """Distance a team must travel from its home base to ``venue``.

    Returns ``0.0`` when either endpoint is unknown (treated as "no measurable
    travel burden") so callers degrade gracefully on unmapped venues/teams.
    """
    home = team_city(team)
    dest = venue_city(venue)
    if home is None or dest is None:
        return 0.0
    return haversine_km(home.lat, home.lon, dest.lat, dest.lon)


def is_interstate(team: str, venue: str | None) -> bool:
    """True if the venue is outside the team's home state."""
    home = team_city(team)
    dest = venue_city(venue)
    if home is None or dest is None:
        return False
    return home.state != dest.state


# --------------------------------------------------------------------------- #
# HGA model
# --------------------------------------------------------------------------- #

# Defaults are fit on 2018-2023 (see ``scripts/eval_ratings.py``) but kept here
# as the documented baseline. ``base_hga`` is the pure crowd/familiarity edge in
# Elo points for a true home game with no travel differential; ``travel_coef``
# converts the away-minus-home travel gap (in km) into extra Elo points.
BASE_HGA_DEFAULT = 30.0
TRAVEL_COEF_DEFAULT = 0.011  # Elo points per km of (away - home) travel
HGA_MIN = -25.0
HGA_MAX = 130.0


def effective_hga(
    home: str,
    away: str,
    venue: str | None,
    *,
    base_hga: float = BASE_HGA_DEFAULT,
    travel_coef: float = TRAVEL_COEF_DEFAULT,
    home_familiarity: float = 1.0,
    hga_min: float = HGA_MIN,
    hga_max: float = HGA_MAX,
) -> float:
    """Venue/travel-aware home-ground advantage in Elo points.

    ``effective_hga = base_hga * home_familiarity + travel_coef * (away_km - home_km)``

    * ``home_familiarity`` in ``[0, 1]`` scales the crowd/familiarity component;
      it should be the share of the home team's recent home games played at this
      venue (1.0 = genuine fortress, ~0 = effectively neutral ground).
    * The travel term rewards the home team when the away side travels further,
      and penalises the home team when it is the one playing away from home.
    """
    home_km = travel_km(home, venue)
    away_km = travel_km(away, venue)
    hga = base_hga * home_familiarity + travel_coef * (away_km - home_km)
    return max(hga_min, min(hga_max, hga))
