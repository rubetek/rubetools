import logging
from typing import List, Tuple
from .shape import Shape

Point = Tuple[float, float, float]
Skeleton = Tuple[int, int]

logger = logging.getLogger(__name__)


class Keypoints(Shape):
    def __init__(self, label: str, points: List[Point], point_names: List[str],
                 skeleton: List[Skeleton], conf: float = 1):
        super(Keypoints, self).__init__(label=label, confidence=conf)
        if len(points) != len(point_names):
            raise ValueError('Number of points and their names are not equal.')

        self._points = points
        self._point_names = point_names
        self._skeleton = skeleton

    def __repr__(self):
        return "<{}, class: {}>".format(self.__class__.__name__, self.label)

    @property
    def points(self):
        return self._points

    @property
    def point_names(self):
        return self._point_names

    @property
    def skeleton(self):
        return self._skeleton

    @classmethod
    def from_polygon(cls, polygon):
        if not isinstance(polygon, Polygon):
            raise ValueError('Incorrect input type.')

        points = [(p[0], p[1], 2) for p in polygon.points]
        point_names = ['point{}'.format(i+1) for i in range(len(points))]
        skeleton = [(i, i+1) for i in range(len(points) - 1)]
        skeleton.append((0, len(points) - 1))
        label = polygon.label
        conf = polygon.confidence

        return cls(label=label, points=points, point_names=point_names, skeleton=skeleton, conf=conf)

    def __eq__(self, other):
        if not isinstance(other, Keypoints):
            return False

        if self.label != other.label or self.points != other.points \
                or self._point_names != other.point_names or self.skeleton != other.skeleton:
            return False

        return True

from .polygon import Polygon