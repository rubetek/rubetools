import numpy as np
import cv2


class Detection(object):
    """
    This class represents a bounding box detection for a single image.

    Parameters
    ----------
    tlbr : array_like
        Bounding box in format `(min x, min y, max x, max y)`.
    class_id : int
        Class id
    confidence : float
        Detector confidence score.

    """

    def __init__(self, tlbr, class_id, confidence):
        for coord in tlbr:
            if coord < 0:
                raise ValueError('Coordinate value should be greater then 0.')
        if tlbr[0] >= tlbr[2] or tlbr[1] >= tlbr[3]:
            raise ValueError('xmax(ymax) should be greater then xmin(ymin).')

        self.tlbr = np.asarray(tlbr, dtype=np.float16)
        self.class_id = int(class_id)
        self.confidence = float(confidence)

    def __repr__(self):
        return "<Detection(class_id: {}, conf: {})>".format(self.class_id, self.confidence)

    @classmethod
    def from_tlwh(cls, tlwh, label, confidence):
        tlbr = np.asarray((tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]), dtype=np.float)
        return cls(tlbr, label, float(confidence))

    def to_tlwh(self):
        """Convert bounding box to format `(x, y, w, h)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlbr.copy()
        ret[2] = ret[2] - ret[0]
        ret[3] = ret[3] - ret[1]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlbr.copy()
        center_x = (ret[0] + ret[2]) / 2
        center_y = (ret[1] + ret[3]) / 2
        ratio = (ret[2] - ret[0])/(ret[3] - ret[1])
        height = ret[3] - ret[1]
        return center_x, center_y, ratio, height

    def get_center(self):
        ret = self.tlbr.copy()
        return (ret[0] + ret[2]) / 2, (ret[1] + ret[3]) / 2

    def __key(self):
        return self.class_id, self.confidence

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if not isinstance(other, Detection):
            return self.__key() == other.__key()

        return sum(self.tlbr == other.tlbr) == 4 and self.class_id == other.class_id \
               and self.confidence == other.confidence

    def _intersect(self, box):
        def _interval_overlap(interval_a, interval_b):
            x1, x2 = interval_a
            x3, x4 = interval_b

            if x3 < x1:
                if x4 < x1:
                    return 0
                else:
                    return min(x2, x4) - x1
            else:
                if x2 < x3:
                    return 0
                else:
                    return min(x2, x4) - x3
        intersect_w = _interval_overlap([self.tlbr[0], self.tlbr[2]], [box[0], box[2]])
        intersect_h = _interval_overlap([self.tlbr[1], self.tlbr[3]], [box[1], box[3]])

        intersect = intersect_w * intersect_h
        return intersect

    def get_iou(self, tlbr):
        """
        Get IOU metric value with tlbr object
        :param tlbr: Bounding box in format `(min x, min y, max x, max y)`
        :return: float
        """
        intersect = self._intersect(tlbr)

        w1, h1 = self.tlbr[2] - self.tlbr[0], self.tlbr[3] - self.tlbr[1]
        w2, h2 = tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]

        union = w1 * h1 + w2 * h2 - intersect
        return float(intersect) / union

    def draw(self, image, labels, colors, print_labels=True):
        image_h, image_w, _ = image.shape

        color = colors[int(self.class_id) % len(colors)]
        xmin, ymin, xmax, ymax = self.tlbr
        f = lambda x: x < 1
        if sum(list(map(f, self.tlbr))) == 4:
            xmin = int(image_w * xmin)
            xmax = int(image_w * xmax)
            ymin = int(image_h * ymin)
            ymax = int(image_h * ymax)
        else:
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        if print_labels:
            title = "{} : {}%".format(labels[self.class_id], int(100*self.confidence))
            cv2.putText(image, title, (xmin, ymin - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.0007 * image_h, color, 2)

        return image

    def to_json(self):
        """
        Serialize Detection to JSON object
        """
        data = {'tlbr': self.tlbr.tolist(),
                'class_id': self.class_id,
                'confidence': self.confidence}
        return data

    @classmethod
    def from_json(cls, json_data=None):
        """
        Load Detection object from JSON file or object
        :param json_data: JSON object
        :return: Detection object
        """
        instance = cls(tlbr=np.array(json_data['tlbr']),
                       class_id=json_data['class_id'],
                       confidence=json_data['confidence'])
        return instance
