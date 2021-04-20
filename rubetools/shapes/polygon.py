import numpy as np
import logging
from typing import List, Tuple
from .shape import Shape

Point = Tuple[float, float]

logger = logging.getLogger(__name__)


class Polygon(Shape):
    """
    Polygon shape of annotation object
    """

    def __init__(self, label: str, points: List[Point], conf: float = 1):
        """
        Constructor
        :param label: object's class name
        :param points: List of object border pairs (x,y)
        """
        super(Polygon, self).__init__(label=label, confidence=conf)
        self._points = points

    def __repr__(self):
        return "<{}, class: {}>".format(self.__class__.__name__, self.label)

    @property
    def points(self):
        return self._points

    @classmethod
    def from_box(cls, box):
        """
        Convert HBox annotation from Polygon annotation
        :param box:
        :return:
        """
        if not isinstance(box, HBox):
            raise ValueError('Incorrect input type.')

        points = [(box.box[0], box.box[1]), (box.box[2], box.box[1]),
                  (box.box[2], box.box[3]), (box.box[0], box.box[3])]

        conf = box.confidence

        return cls(label=box.label, points=points, conf=conf)

    def from_keypoints(self, keypoints):
        if not isinstance(keypoints, Keypoints):
            raise ValueError('Incorrect input type.')
        raise NotImplemented

    @property
    def area(self):
        """
        Calculates area of polygon
        :return:
        """
        x, y = map(np.array, zip(*self._points))
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def __eq__(self, other):
        if not isinstance(other, Polygon):
            return False

        if self.label != other.label or self.points != other.points:
            return False

        return True

from .hbox import HBox
from .keypoints import Keypoints