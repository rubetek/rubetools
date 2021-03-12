from __future__ import annotations

import numpy as np
import os
import shutil
from datetime import datetime
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Type
from ..annotation import Annotation
from ..detection import Detection
from ..shapes.hbox import HBox
from ..shapes.polygon import Polygon
from ..eda import EDA
from ..utils import Image, colorstr

Point = Tuple[np.float16]


class FormatBase(ABC):
    """
    Abstract class - base class for all implemented annotations formats
    """
    def __init__(self, ann_path: str = None, img_path: str = None,
                 selected_labels: List[str] = None, is_load_empty_ann: bool = True, **kwargs):
        if selected_labels is None:
            selected_labels = []
        self._ann_dir = ann_path
        self._img_dir = img_path
        self._labels_stat = {}
        self._annotations = []

        self.log = logging.getLogger(__name__)
        if ann_path:
            self._load(selected_labels, is_load_empty_ann=is_load_empty_ann)

        self._eda = EDA(annotations=self.annotations)

    def __repr__(self):
        return '<{}, annotations: {}, labels: {}>'.format(str(self.__class__.__name__), len(self._annotations),
                                                          len(self._labels_stat))

    @abstractmethod
    def _load(self, labels: List[str], **kwargs):
        """
        Load annotations
        :param labels: list of class labels. If None, load all seen labels
        :param kwargs:
        :return:
        """
        if not (isinstance(self._img_dir, str) and isinstance(self._ann_dir, str)):
            raise ValueError('Incorrect input paths, is not str.')

        if not (os.path.exists(self._img_dir) and os.path.exists(self._ann_dir)):
            raise FileNotFoundError('Input paths are not exist.')

    @abstractmethod
    def save(self, save_dir: str = None, is_save_images: bool = False, **kwargs):
        """
        Save annotations
        :param save_dir: save directory path. If save_dir is None, write annotations to current directory
        :param is_save_images: if True - save images, otherwise - do not save image files
        :param kwargs:
        :return:
        """
        raise NotImplemented

    @property
    def annotation_directory(self):
        return self._ann_dir

    @property
    def image_directory(self):
        return self._img_dir

    @property
    def labels(self):
        return list(self._labels_stat.keys())

    @property
    def labels_stat(self):
        return self._labels_stat

    @property
    def annotations(self):
        return self._annotations

    @property
    def eda(self):
        return self._eda

    def convert_to(self, ann_type: Type[FormatBase], save_path: str, is_save_images: bool = False):
        """
        Convert annotations from to any format to one (Yolo, PascalVOC, ...)
        and saving the annotations received when converting to the save_path folder
        :param ann_type: class object, to which you want to convert the annotations
        :param save_path: annotation save path
        :param is_save_images: if True - save images, otherwise - do not save image files
        :return:
        """
        instance = ann_type.init_from_annotations(self._annotations)
        instance.save(save_dir=save_path, is_save_images=is_save_images)

    @classmethod
    def init_from_annotations(cls, annotations: List[Annotation]):
        """
        Initialisation of annotations by universal presentation and seen label dictionary
        :param annotations: list of Annotation objects
        :return:
        """
        instance = cls()
        instance._annotations = annotations
        instance._labels_stat = {}
        for ann in instance.annotations:
            for obj in ann:
                obj_name = obj.label
                if obj_name in instance._labels_stat:
                    instance._labels_stat[obj_name] += 1
                else:
                    instance._labels_stat[obj_name] = 1

        instance._eda = EDA(annotations=instance.annotations)
        return instance

    @classmethod
    def from_detections(cls, detections: List[List[Detection]], labels: List[str], img_paths: List[str]):
        """
        Initialisation of annotations from list of Detection
        :param detections: Detections for every object in list
        :param labels: list of class labels
        :param img_paths: list of image paths
        :return:
        """
        if not isinstance(detections[0], list) or isinstance(img_paths, str):
            raise ValueError('Incorrect input data.')

        instance = cls()

        for img_dets, img_path in zip(detections, img_paths):
            img_shape = Image.get_shape(img_path=img_path)
            ann = Annotation(img_path=img_path, width=img_shape[1], height=img_shape[0], depth=img_shape[2])

            for det in img_dets:
                obj_name = labels[det.class_id]
                if obj_name in instance._labels_stat:
                    instance._labels_stat[obj_name] += 1
                else:
                    instance._labels_stat[obj_name] = 1
                x_min, y_min, x_max, y_max = det.tlbr
                box = HBox(obj_name, x_min, y_min, x_max, y_max)
                ann.add(box)

            if len(ann) > 0:
                instance._annotations += [ann]

        instance._eda = EDA(annotations=instance.annotations)
        return instance

    @classmethod
    def from_polygons(cls, polygons, labels: List[str], img_paths: List[str]):
        """
        Initialisation of annotations from list of polygons
        :param polygons: Polygons for every object in list
        :param labels: list of class labels
        :param img_paths: list of image paths
        :return:
        """
        raise Exception('Not implemented.')

    def rename_classes(self, rename_dict: Dict[str, str]):
        """
        Rename class labels for every annotation
        :param rename_dict: class name renaming dictionary. key: old class name, value: new class name
        :return:
        """
        assert len(self._annotations) > 0

        obj_rename_count = 0
        ann_rename_count = 0

        new_seen_labels = {}
        for i in range(len(self._annotations)):
            cur_obj_counter = 0
            for obj in self._annotations[i]:
                label = obj.label
                if label in list(rename_dict.keys()):
                    obj.label = rename_dict[label]
                    cur_obj_counter += 1

                if obj.label in new_seen_labels:
                    new_seen_labels[obj.label] += 1
                else:
                    new_seen_labels[obj.label] = 1

            if cur_obj_counter > 0:
                obj_rename_count += cur_obj_counter
                ann_rename_count += 1

        self._labels_stat = new_seen_labels
        self.log.info("Renamed {} objects in {} annotations.".format(colorstr('cyan', 'bold', obj_rename_count),
                                                                     colorstr('cyan', 'bold', ann_rename_count)))

    def delete_objects(self, min_size=(20, 20), classes=None):
        """
        Delete objects in annotations by min_size or class labels
        :param min_size:
        :param classes:
        :return:
        """
        raise NotImplemented

    def slice_images(self, img_save_dir: str, shape: Tuple[int, int] = (416, 416),
                     start_w: int = 0, start_h: int = 0, step: int = 416, overlap: float = 0.5):
        """
        Slice too large image into several smaller then min_shape
        :param img_save_dir: save image directory path (annotations do not saving, for saving use method save())
        :param shape: split image shape
        :param start_w: start x point
        :param start_h: start y point
        :param step: step for splitting image
        :param overlap: square overlap level for objects in annotation.
        if smaller then value, object delete from annotation
        :return:
        """
        num_base_annotations = len(self.annotations)
        if num_base_annotations == 0:
            self.log.info('Annotation list is empty. Nothing to slicing.')
            return

        os.makedirs(img_save_dir, exist_ok=True)
        time_now = datetime.strftime(datetime.today(), '%m-%d-%H-%M')
        target_dir = os.path.join(img_save_dir, 'images_{}_{}_{}'.format(shape[0], shape[1], time_now))
        os.makedirs(target_dir, exist_ok=True)

        cut_anns = []
        new_seen_labels = {}
        for ann in tqdm(self.annotations):
            img = Image.get_buffer(img_path=ann.img_path)
            h, w = img.shape[0:2]

            end_w = 0
            end_h = 0
            counter = 0
            while end_w < w:
                while end_h < h:
                    # save split image
                    end_w = start_w + shape[0]
                    end_h = start_h + shape[1]
                    cut_img = img[start_h:end_h, start_w:end_w]

                    # skip small images
                    if cut_img.shape[0]*cut_img.shape[1] < overlap * shape[0] * shape[1]:
                        break

                    # save crop image
                    filename, ext = os.path.splitext(os.path.basename(ann.img_path))
                    img_save_path = os.path.join(target_dir, filename + '_' + str(counter) + ext)
                    Image.save_buffer(save_path=img_save_path, buffer=cut_img)

                    # build split ann
                    filename = img_save_path
                    height, width, depth = cut_img.shape
                    cut_ann = Annotation(img_path=filename, width=width, height=height, depth=depth)
                    for obj in ann.objects:
                        if type(obj) == HBox:
                            x_min = min(max(obj.box[0] - start_w, 0), end_w-start_w)
                            x_max = max(min(obj.box[2] - start_w, end_w-start_w), 0)
                            y_min = min(max(obj.box[1] - start_h, 0), end_h-start_h)
                            y_max = max(min(obj.box[3] - start_h, end_h-start_h), 0)

                            obj_square = (obj.box[2] - obj.box[0])*(obj.box[3] - obj.box[1])
                            if overlap * obj_square <= (x_max - x_min)*(y_max - y_min) <= obj_square:
                                box = HBox(obj.label, x_min, y_min, x_max, y_max)
                                cut_ann.add(box)

                                if obj.label in new_seen_labels:
                                    new_seen_labels[obj.label] += 1
                                else:
                                    new_seen_labels[obj.label] = 1

                        if type(obj) == Polygon:
                            raise NotImplemented
                    cut_anns.append(cut_ann)

                    counter += 1
                    start_h += step
                    if start_h > h - shape[1]:
                        start_h = h - shape[1]

                start_w += step
                start_h = 0
                end_h = 0
                if start_w > w - shape[0]:
                    start_w = w - shape[0]

        self._annotations = cut_anns
        self._ann_dir = None
        self._img_dir = target_dir
        self._labels_stat = new_seen_labels
        self._eda = EDA(annotations=self.annotations)
        self.log.info("Split {} fragments from {} images.".format(colorstr('cyan', 'bold', len(self._annotations)),
                                                                  colorstr('cyan', 'bold',num_base_annotations)))

    def split(self, ratio: Tuple[float, float] = (0.8, 0.2), seed: int = 128):
        """
        Split Dataset into sub datasets. For example train and test.
        :param ratio: ration between sub datasets
        :param seed: random number generator state
        :return:
        """
        if sum(ratio) != 1:
            ValueError("Ratio coefficient's sum should equal 1.")

        if len(self._annotations) == 0:
            return []

        np.random.seed(seed)
        np.random.shuffle(self._annotations)
        np.random.seed(None)

        train_ratio = ratio[0]
        separator = int(len(self._annotations) * train_ratio)

        train_data = self._annotations[0: separator]
        val_data = self._annotations[separator:]

        train_instance = self.init_from_annotations(annotations=train_data)
        test_instance = self.init_from_annotations(annotations=val_data)

        return train_instance, test_instance

    def sparse(self, step: int):
        """
        Thin out the selection with every step element
        :param step: decimation step
        :return:
        """
        num_ann = len(self._annotation)
        assert num_ann > 0
        self._annotation = self._annotation[0:num_ann:step]
        self._eda = EDA(annotations=self.annotations)

    def join(self, new_annotations: List[List[Annotation]]):
        """
        Join annotation files of one image
        :param new_annotations: new added list of annotations
        :return:
        """
        raise NotImplemented

    @staticmethod
    def _get_target_by_source_filenames(source_dir: str, target_dir: str, save_dir: str, move: bool = True):
        """
        Move or copy files from target directory if exists in source
        :param source_dir: dir, by which we check files for transport
        :param target_dir: dir, from which we select files for transport
        :param save_dir: dir, where will save result
        :param move: bool, move files if True, copy files if False
        :return:
        """
        if not os.path.exists(source_dir) or not os.path.exists(target_dir):
            raise ValueError('Empty image or annotation directory.')

        dir_out = os.path.join(save_dir, os.path.basename(target_dir))
        if move:
            dir_out += '_moved'
        else:
            dir_out += '_copied'
        os.makedirs(dir_out, exist_ok=True)

        source_files = os.listdir(source_dir)
        target_files = os.listdir(target_dir)

        for s_file_name in tqdm(source_files):
            try:
                source_name = os.path.splitext(s_file_name)[0]
                for t_file_name in target_files:
                    name, ext = os.path.splitext(t_file_name)
                    if name == source_name:
                        path_in = os.path.join(target_dir, t_file_name)
                        path_out = os.path.join(dir_out, t_file_name)
                        if move:
                            shutil.move(path_in, path_out)
                        else:
                            shutil.copy(path_in, path_out)
                        break
            except Exception as exc:
                continue

    @staticmethod
    def get_images_by_annotations(ann_dir: str, img_dir: str, save_dir: str, move: bool = True):
        """
        Move or copy images if exists annotations by the same names
        :param ann_dir: annotation directory path
        :param img_dir: image directory path
        :param save_dir: dir, where will save result
        :param move: bool, move files if True, copy files if False
        :return:
        """
        FormatBase._get_target_by_source_filenames(source_dir=ann_dir, target_dir=img_dir, save_dir=save_dir, move=move)

    @staticmethod
    def get_annotations_by_images(ann_dir: str, img_dir: str, save_dir: str, move: bool = True):
        """
        Move or copy annotations if exists images by the same names
        :param ann_dir: annotation directory path
        :param img_dir: image directory path
        :param save_dir: dir, where will save result
        :param move: bool, move files if True, copy files if False
        :return:
        """
        FormatBase._get_target_by_source_filenames(source_dir=img_dir, target_dir=ann_dir, save_dir=save_dir, move=move)
