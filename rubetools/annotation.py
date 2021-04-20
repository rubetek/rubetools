import os
from .shapes.hbox import HBox
from .shapes.polygon import Polygon
from .shapes.keypoints import Keypoints
from .object import Object


class Annotation:
    """
    Universal presentation (meta format) of image annotation object
    """
    def __init__(self, img_path: str, width: float, height: float, depth=None):
        self._img_path = img_path
        self._width = width
        self._height = height
        self._depth = depth

        self._objects = []

    def __repr__(self):
        return '<{}, img: {}>'.format(self.__class__.__name__, os.path.basename(self.img_path))

    @property
    def img_path(self):
        return self._img_path

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def depth(self):
        return self._depth

    @property
    def objects(self):
        return self._objects

    def add(self, obj):
        """
        Add new object one of shape type
        :param obj:
        :return:
        """
        if isinstance(obj, Object):
            self._objects.append(obj)
        elif isinstance(obj, HBox):
            new_obj = Object(hbox=obj)
            self._objects.append(new_obj)
        elif isinstance(obj, Polygon):
            new_obj = Object(polygon=obj)
            self._objects.append(new_obj)
        elif isinstance(obj, Keypoints):
            new_obj = Object(keypoints=obj)
            self._objects.append(new_obj)
        else:
            raise ValueError("Unsupported object's type")

    def __len__(self):
        return len(self.objects)

    def __iter__(self):
        return iter(self._objects)

    def __eq__(self, other):
        if not isinstance(other, Annotation):
            return False

        if self.width != other.width or self.height != other.height or self.depth != other.depth:
            return False

        if self.img_path != other.img_path:
            return False

        if len(self.objects) != len(other.objects):
            return False

        for i in range(len(self.objects)):
            if len(self.objects[i]) != len(other.objects[i]):
                return False

        return True
