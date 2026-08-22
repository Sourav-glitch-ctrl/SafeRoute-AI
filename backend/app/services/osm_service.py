import httpx


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

HEADERS = {
    "User-Agent": "SafeRoute-AI/1.0"
}


def get_road_features(
    latitude: float,
    longitude: float,
    radius_m: int = 100,
) -> dict:
    """
    Get nearby road-infrastructure features from OpenStreetMap.

    Returns binary indicators compatible with the SafeRoute
    feature-engineering pipeline.
    """

    query = f"""
    [out:json][timeout:25];

    (
        node(around:{radius_m},{latitude},{longitude})
            ["highway"="traffic_signals"];

        node(around:{radius_m},{latitude},{longitude})
            ["highway"="stop"];

        node(around:{radius_m},{latitude},{longitude})
            ["railway"];

        node(around:{radius_m},{latitude},{longitude})
            ["highway"="crossing"];

        way(around:{radius_m},{latitude},{longitude})
            ["junction"];

    );

    out tags;
    """

    try:
        response = httpx.get(
            OVERPASS_URL,
            params={"data": query},
            headers=HEADERS,
            timeout=30.0,
        )

        response.raise_for_status()

        data = response.json()

    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"OpenStreetMap request failed: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            f"Invalid OpenStreetMap response: {exc}"
        ) from exc

    traffic_signal = 0
    crossing = 0
    junction = 0
    railway = 0
    stop = 0

    for element in data.get("elements", []):

        tags = element.get("tags", {})

        highway = tags.get("highway")

        if highway == "traffic_signals":
            traffic_signal = 1

        if highway == "crossing":
            crossing = 1

        if highway == "stop":
            stop = 1

        if "railway" in tags:
            railway = 1

        if "junction" in tags:
            junction = 1

    near_road_infrastructure = int(
        traffic_signal
        or crossing
        or junction
        or railway
        or stop
    )

    return {
        "Traffic_Signal": traffic_signal,
        "Crossing": crossing,
        "Junction": junction,
        "Railway": railway,
        "Stop": stop,
        "NearRoadInfrastructure": near_road_infrastructure,
    }