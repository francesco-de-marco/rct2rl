import math

def centermost_coord(coordinates):
    assert len(coordinates) > 0
    mean_x = sum(x for x, y in coordinates) / len(coordinates)
    mean_y = sum(y for x, y in coordinates) / len(coordinates)

    def distance(coord, mean_coord):
        return math.sqrt((coord[0] - mean_coord[0]) ** 2 + (coord[1] - mean_coord[1]) ** 2)

    mean_coord = (mean_x, mean_y)
    centermost_coord = min(coordinates, key=lambda coord: distance(coord, mean_coord))
    return centermost_coord