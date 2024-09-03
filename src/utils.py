import math

from pyproj import Transformer
from shapely import geometry


DEFAULT_EPSG = "EPSG:4326"


def determine_zone(longitude: float) -> int:
    """Determine the UTM zone from a longitude value

    Parameters
    ----------
    longitude : float
        Value describing the longitude. Negative values denote west of the prime meridian, and
        positive values denote east of the prime meridian.

    Returns
    -------
    int
        The UTM zone
    """
    return math.ceil((longitude + 180) / 6)


def determine_hemisphere(latitude: float) -> str:
    """Determine whether latitude in the northern or southern hemisphere

    Parameters
    ----------
    latitude : float
        Value describing the latitude. Negative values denote south of the equator, and
        positive values denote north of the equator.

    Returns
    -------
    str
        "north" or "south
    """
    if latitude > 0:
        return "north"
    return "south"


def determine_crs_code(longitude: float, latitude: float) -> str:
    """Determine the coordinate reference system code

    Parameters
    ----------
    longitude : float
        Value describing the longitude. Negative values denote west of the prime meridian, and
        positive values denote east of the prime meridian.
    latitude : float
        Value describing the latitude. Negative values denote south of the equator, and
        positive values denote north of the equator.

    Returns
    -------
    str
        The coordinate reference system code in meters
    """
    zone = determine_zone(longitude)
    hemisphere = determine_hemisphere(latitude)
    epsg = 32600 + zone if hemisphere == "north" else 32700 + zone
    return str(epsg)


def convert_degree_to_meters(longitude: float, latitude: float) -> tuple[float, float, str]:
    """Convert a longitude and latitude point into the easting and northing values in meters

    Parameters
    ----------
    longitude : float
        Value describing the longitude. Negative values denote west of the prime meridian, and
        positive values denote east of the prime meridian.
    latitude : float
        Value describing the latitude. Negative values denote south of the equator, and
        positive values denote north of the equator.

    Returns
    -------
    tuple[float, float, str]
        [0] The easting value in meters
        [1] The northing value in meters
        [2] The epsg_code in UTM
    """
    epsg_code_utm = f"EPSG:{determine_crs_code(longitude, latitude)}"
    transformer = Transformer.from_crs(DEFAULT_EPSG, epsg_code_utm, always_xy=True)
    lon_m, lat_m = transformer.transform(longitude, latitude)
    return lon_m, lat_m, epsg_code_utm


def convert_meters_to_degree(
    lon_m: float, lat_m: float, epsg_code_utm: str
) -> tuple[float, float]:
    """Convert the easting and northing values in meters into degrees longitude and latitude

    Parameters
    ----------
    lon_m : float
        Easting value in meters
    lat_m : float
        Northing value in meters
    epsg_code_utm : str
        The epsg_code in UTM

    Returns
    -------
    tuple[float, float]
        [0] The longitude value in degrees
        [1] The latitude value in degrees
    """
    transformer = Transformer.from_crs(epsg_code_utm, DEFAULT_EPSG, always_xy=True)
    return transformer.transform(lon_m, lat_m)


def create_polygon_from_list_tuple(list_tuple_coords: list[tuple]) -> geometry.Polygon:
    """Create a Polygon object from a list of coordinates

    Parameters
    ----------
    list_tuple_coords : list[tuple]
        List of coordinates, ie [(lon1, lat1), (lon2, lat2), (lon3, lat3)]

    Returns
    -------
    geometry.Polygon
        A Polygon object representative of the given coordinates
    """
    list_points = [geometry.Point(coord) for coord in list_tuple_coords]
    return geometry.Polygon([[pt.x, pt.y] for pt in list_points])
