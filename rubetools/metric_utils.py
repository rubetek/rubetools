import os
from typing import List, Tuple
from shapely.geometry import Polygon
import numpy as np

from .annotation import Annotation
from .shapes.hbox import HBox
from .shapes.polygon import Polygon as PolygonShape

Point = Tuple[float, float]


def get_list_objects(list_annotations: List[Annotation], class_name: str, is_sort_by_confidence: bool = False):
    list_objects = []
    list_confidances = []

    # создаем список объектов: (image_name, [(point1, point2, point3, point4)])
    for ann in list_annotations:
        img_name = os.path.basename(ann.img_path)

        for obj in ann.objects:
            if obj.label == class_name:
                if obj.hbox is not None:
                    list_confidances.append(obj.hbox.confidence)
                    transform_obj = PolygonShape.from_box(obj.hbox)
                    list_objects.append((img_name, transform_obj.points))
                elif obj.polygon is not None:
                    list_confidances.append(obj.polygon.confidence)
                    list_objects.append((img_name, obj.polygon.points))

    # в случае необходимости, сортируем объекты по confidence
    if is_sort_by_confidence:
        indices = np.flip(np.argsort(list_confidances))
        list_objects = [list_objects[i] for i in indices]

    return list_objects


def get_iou(box_1: List[Tuple], box_2: List[Tuple], iou_level: float = 0.5):
    b1 = Polygon(box_1)
    b2 = Polygon(box_2)

    # вычисление метрики IoU между двумя боксами, заданными полигонами
    iou = b1.intersection(b2).area / b1.union(b2).area

    return iou


class ConfusionMatrix:
    def __init__(self, targets: List[Tuple[str, List[Point]]], preds: List[Tuple[str, List[Point]]]):
        self._targets = targets
        self._preds = preds

        self._ious_matrix = self._create_ious_matrix()

    @property
    def targets(self):
        return self._targets

    @property
    def preds(self):
        return self._preds

    def _create_ious_matrix(self):
        ious_matrix = np.zeros(
            shape=(len(self.targets), len(self.preds)), dtype=np.float16)

        for i, target in enumerate(self.targets):
            for j, pred in enumerate(self.preds):
                # если истиный объект (target) и предсказанный обект (pred) принадлежат разным изображениям,
                # не считаем IoU, поле в матрице остается равным 0
                if target[0] != pred[0]:
                    continue
                # в противном случае считаем IoU, если вычисленная метрика больше указанного порога,
                # соответсвующему элемкнту матрицы присваиваем значение 1
                else:
                    iou = get_iou(target[1], pred[1])
                    ious_matrix[i, j] = iou

        return ious_matrix

    def _create_confusion_ious_matrix(self, iou_level: float = 0.5):

        indices = self._ious_matrix > iou_level
        confusion_ious_matrix = self._ious_matrix * indices

        return confusion_ious_matrix

    def _create_confusion_matrix(self, iou_level: float = 0.5):

        confusion_ious_matrix = self._create_confusion_ious_matrix(iou_level)
        confusion_matrix = np.zeros_like(confusion_ious_matrix, dtype=np.bool_)

        for i in range(confusion_ious_matrix.shape[1]):
            _max = np.amax(confusion_ious_matrix[:, i])
            if _max > 0:
                index = confusion_ious_matrix[:, i].argmax(axis=0)
                confusion_ious_matrix[index, :] = 0
                confusion_matrix[index, i] = 1

        return confusion_matrix

    def calc_precision_recall(self, iou_level: float = 0.5):
        # если матрица не создана, создаем матрицу confusion_matrix с заданным порогом IoU (iou_level)
        confusion_matrix = self._create_confusion_matrix(iou_level)

        # вычисляем TP, FP, FN
        tp = np.sum(np.sum(confusion_matrix, axis=1) > 0)
        fp = np.sum(np.sum(confusion_matrix, axis=0) == 0)
        fn = np.sum(np.sum(confusion_matrix, axis=1) == 0)

        # вычисляем precision, recall, F1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)

        return precision, recall, f1

    def calc_ap(self, iou_level: float = 0.5):
        # создаем матрицу confusion_matrix с заданным порогом IoU (iou_level)
        confusion_matrix = self._create_confusion_matrix(iou_level)

        # подготовительные действия для вычисления кривой pr/rec
        axis_1 = np.cumsum(confusion_matrix, axis=1)
        axis_0 = np.sum(confusion_matrix, axis=0)

        tp_history = np.sum((axis_1 > 0), axis=0)
        fn_history = np.sum((axis_1 == 0), axis=0)
        fp_history = np.cumsum(axis_0 == 0)

        precision_history = tp_history / (tp_history + fp_history)
        recall_history = tp_history / (tp_history + fn_history)

        # интерполируем кривую precision/recall на 101 точку
        m_reccall = np.concatenate(([0.], recall_history))
        m_precision = np.concatenate(([1.], precision_history))

        m_precision = np.flip(np.maximum.accumulate(np.flip(m_precision)))

        x = np.linspace(0, recall_history[-1], 101)
        average_precision = np.trapz(np.interp(x, m_reccall, m_precision), x)

        return average_precision

    def calc_ap_05_95(self):
        ious_level = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        ap = []

        for iou_level in ious_level:
            ap.append(self.calc_ap(iou_level))

        ap_05_95 = sum(ap) / len(ap)

        return ap_05_95
