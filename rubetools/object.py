from .shapes.hbox import HBox
from .shapes.keypoints import Keypoints
from .shapes.polygon import Polygon


class Object:
    def __init__(self, hbox: HBox = None, keypoints: Keypoints = None, polygon: Polygon = None):
        self._hbox = hbox
        self._keypoints = keypoints
        self._polygon = polygon

        shapes = {}
        for name, value in vars(self).items():
            if value is not None:
                shapes[name[1:]] = getattr(self, name)
        self._not_none_shapes = shapes

    def __repr__(self):
        return '<{}, shapes: {}>'.format(self.__class__.__name__, ' '.join(self._not_none_shapes.keys()))

    @property
    def hbox(self):
        return self._hbox

    @property
    def keypoints(self):
        return self._keypoints

    @property
    def polygon(self):
        return self._polygon

    @property
    def label(self):
        attrs = vars(self)
        for name, value in attrs.items():
            if value is not None:
                return getattr(self, name).label

        return None

    def join(self, obj):
        """
        Append new objects with different shapes
        :param obj: new added Object instance
        :return:
        """
        for new_shape_name, new_shape in obj:
            for cur_shape_name, cur_shape in self._not_none_shapes.items():
                if cur_shape_name == new_shape_name and new_shape != cur_shape:
                    setattr(self, '_' + cur_shape_name, new_shape)

    @staticmethod
    def get_available_shapes():
        attrs = vars(Object())
        return [shape[1:] for shape in attrs.keys()][:-1]

    def __len__(self):
        return len(self._not_none_shapes)

    def __iter__(self):
        return iter(self._not_none_shapes.items())
