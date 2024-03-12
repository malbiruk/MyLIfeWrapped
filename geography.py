'''
This module contains helper functions to draw a map using google map static API
'''
import math

import googlemaps
import numpy as np
from matplotlib.transforms import Affine2D


def calculate_zoom_level(x_min: float, x_max: float,
                         y_min: float, y_max: float,
                         image_size: tuple) -> int:
    '''
    calculate optimal zoom level based on coordinate bounds
    '''
    # Google Maps static maps parameters
    tile_size = 256
    max_zoom = 20

    # Calculate the bounding box dimensions
    lat_ratio = math.radians(y_max - y_min) / math.pi
    lon_ratio = math.radians(x_max - x_min) / math.pi

    # Iterate through zoom levels
    for zoom in range(max_zoom, 0, -1):
        # Calculate the number of tiles in both dimensions
        num_tiles_x = 2 ** zoom
        num_tiles_y = 2 ** zoom

        # Calculate pixel dimensions of the bounding box
        pixel_width = tile_size * num_tiles_x * lon_ratio
        pixel_height = tile_size * num_tiles_y * lat_ratio

        # Check if the bounding box fits within the desired image size
        if pixel_width < image_size[0] and pixel_height < image_size[1]:
            return zoom

    # If the loop completes without finding a suitable zoom level, return 0
    return 0


def lat_lng_to_image_coords(lat: float, lng: float,
                            center_lat: float, center_lon: float,
                            zoom: int, img_size: tuple) -> tuple:
    '''
    transfer coords (lattitude, longitude) of one point to
    xy pixel coords of image
    '''
    parallel_multiplier = math.cos(lat * math.pi / 180)
    degrees_pp_x = 360 / math.pow(2, zoom + 8)
    degrees_pp_y = 360 / math.pow(2, zoom + 8) * parallel_multiplier

    x = img_size[0] / 2 + (lng - center_lon) / degrees_pp_x
    y = img_size[1] / 2 - (lat - center_lat) / degrees_pp_y

    return (x, y)


def get_transform_2pts(q1: np.array, q2: np.array,
                       p1: np.array, p2: np.array) -> Affine2D:
    '''
    create transform to transform from q to p,
    such that q1 will point to p1, and q2 to p2
    '''
    ang = (np.arctan((p2 - p1)[1] / (p2 - p1)[0])
           - np.arctan((q2 - q1)[1] / (q2 - q1)[0]))
    s = np.abs(np.sqrt(np.sum((p2 - p1)**2)) / np.sqrt(np.sum((q2 - q1)**2)))
    trans = Affine2D().translate(*-q1).rotate(ang).scale(s).translate(*p1)
    return trans


def generate_map(x_bounds: tuple, y_bounds: tuple,
                 center: tuple, map_size: tuple,
                 api_key: str, output_path: str,
                 return_trans: bool=True,
                 **kwargs) -> None:
    '''
    save google map and (optionally) return transform object
    for plotting

    x_bounds - tuple(x_min, x_max)
    y_bounds - tuple(y_min, y_max)
    center - tuple(lat, lon)
    map_size - tuple(width, height)
    api_key - Google Maps Static API key
    output_path - where to save final image
    return_trans - return trans object for matplotlib
    **kwargs go to googlemaps.Client().static_map() method
    '''
    zoom = calculate_zoom_level(*x_bounds, *y_bounds, map_size)
    gmaps = googlemaps.Client(key=api_key)

    with open(output_path, 'wb') as f:
        for chunk in gmaps.static_map(size=map_size,
                                      center=','.join([str(i) for i in center]),
                                      zoom=zoom,
                                      **kwargs):
            if chunk:
                f.write(chunk)

    if return_trans:
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        lc1 = np.array([x_min, y_min])
        ic1 = np.array(
            lat_lng_to_image_coords(x_min, y_min, *center, zoom, map_size)
        )

        lc2 = np.array([x_max, y_max])
        ic2 = np.array(
            lat_lng_to_image_coords(x_max, y_max, *center, zoom, map_size)
        )
        return get_transform_2pts(lc1, lc2, ic1, ic2)

    return None
