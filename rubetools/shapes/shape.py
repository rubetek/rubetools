from abc import ABC


class Shape(ABC):
    def __init__(self, label: str, confidence: float = 1.0):
        self._label = label
        self._conf = confidence

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        if isinstance(value, str):
            self._label = value
        else:
            raise ValueError('Expected str value.')

    @property
    def confidence(self):
        return self._conf
