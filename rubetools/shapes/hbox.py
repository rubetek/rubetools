import logging
from .shape import Shape

logger = logging.getLogger(__name__)


class HBox(Shape):
    """
    Horizontal box (hbox) shape of annotation object
    """
    def __init__(self, label: str, x_min: float, y_min: float, x_max: float, y_max: float, conf: float = 1.0):
        """
        Constructor
        :param label: object's class name
        :param x_min: top left x coordinate
        :param y_min: top left y coordinate
        :param x_max: bottom right x coordinate
        :param y_max: bottom right y coordinate
        """
        super(HBox, self).__init__(label=label, confidence=conf)
        self._box = [x_min, y_min, x_max, y_max]

    def __repr__(self):
        return '<{}, class: {}>'.format(self.__class__.__name__, self.label)

    @property
    def box(self):
        return self._box

    @property
    def pixel_area(self):
        return round((self._box[2] - self._box[0] + 1) * (self._box[3] - self._box[1] + 1), 2)

    @classmethod
    def from_polygon(cls, polygon):
        if not isinstance(polygon, Polygon):
            raise ValueError('Incorrect input type.')

        points = polygon.points
        label = polygon.label

        x, y = map(list, zip(*points))

        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        conf = polygon.confidence

        return cls(label=label, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, conf=conf)

    def __eq__(self, other):
        if not isinstance(other, HBox):
            return False

        if self.label != other.label or self.box != other.box:
            return False

        return True

from .polygon import Polygon