import os
from glob import glob
from tqdm import tqdm
import datetime
from collections import OrderedDict
import json
from typing import List, Union
from .base import FormatBase
from ..shapes.hbox import HBox
from ..shapes.polygon import Polygon
from ..shapes.keypoints import Keypoints
from ..annotation import Annotation
from ..object import Object
from ..utils import Image


class MSCOCO(FormatBase):
    """
    The MSCOCO annotation has following structure:
    {
        "images": [
            {
                "file_name": ,
                "height": ,
                "width": ,
                "id":
            },
            ...
        ],
        "type": "instances",
        "annotations": [
            {
                "segmentation": [],
                "area": ,
                "iscrowd": ,
                "image_id": ,
                "bbox": [],
                "category_id": ,
                "id": ,
                "ignore": ,
                "keypoints": ,
                "num_keypoints":
            },
            ...
        ],
        "categories": [
            {
                "supercategory": ,
                "id": ,
                "name": ,
                "keypoints": ,
                "skeleton":
            },
            ...
        ]
    }
    """

    def _load(self, labels: List[str] = None, is_load_empty_ann: bool = True, ground_truth: str = None):
        """
        Load MSCOCO format annotations
        :param labels: list of class labels. If None, load all seen labels
        :param is_load_empty_ann: if True - load annotations with empty objects images, otherwise - skip.
        :return:
        """
        img_paths = None
        if self._img_path is not None and isinstance(self._img_path, str) and os.path.isdir(self._img_path):
            img_paths = [f for f in glob(self._img_path + '/*', recursive=True) if os.path.isfile(f)]

        if self._ann_path is not None and isinstance(self._ann_path, str) and os.path.isfile(self._ann_path):
            with open(self._ann_path, 'r') as f_ann:
                dataset = json.load(f_ann)
        else:
            raise FileNotFoundError(
                'Incorrect initialization annotation path.')

        if ground_truth is None:
            categories = dataset['categories']
            images = dataset['images'] if img_paths is None else img_paths
            annotations = dataset['annotations']
        else:
            if ground_truth is not None and isinstance(ground_truth, str) and os.path.isfile(ground_truth):
                with open(ground_truth, 'r') as f_ann:
                    gt = json.load(f_ann)
            else:
                raise FileNotFoundError('Incorrect initialization ground truth annotation path.')

            categories = gt['categories']
            images = gt['images'] if img_paths is None else img_paths
            annotations = dataset
            dataset = {'images': gt['images']}

        # creating a dictionary of object categories in a dataset: {id_class: name_class}
        classes_dict = dict()
        for class_obj in categories:
            class_obj['keypoints'] = class_obj.get('keypoints', 0)
            class_obj['skeleton'] = class_obj.get('skeleton', 0)
            classes_dict[class_obj['id']] = [class_obj['name'], class_obj['keypoints'], class_obj['skeleton']]

        for img in tqdm(images):
            if isinstance(img, str):
                width, height, depth = Image.get_shape(img)
                img_path = img
                ann_instances = list(filter(lambda t: t['file_name'] == os.path.basename(img), dataset['images']))
                if len(ann_instances) == 0:
                    continue
                img_id = ann_instances[0]['id']
            else:
                width = int(img['width'])
                height = int(img['height'])
                if 'depth' in img and img['depth'] is not None:
                    depth = int(img['depth'])
                else:
                    depth = 3
                img_path = img['file_name']
                img_id = img['id']

            ann = Annotation(img_path=img_path, width=width, height=height, depth=depth)
            for one_obj in annotations:
                # check if the next object belongs to our image
                if one_obj['image_id'] == img_id:
                    label = classes_dict[one_obj["category_id"]][0]
                    kp_names = classes_dict[one_obj['category_id']][1]
                    sk = classes_dict[one_obj['category_id']][2]
                    if labels is not None and len(labels) > 0 and label not in labels:
                        continue
                    else:
                        confidence = 1 if one_obj.get('score') is None else one_obj.get('score')
                        segm = one_obj.get('segmentation')
                        kp = one_obj.get('keypoints')
                        bb = one_obj.get('bbox')

                        polygon = box = keypoints = None
                        is_object = False
                        if segm is not None and segm:
                            points = [(segm[0][i], segm[0][i+1]) for i in range(0, len(segm[0]), 2)]
                            polygon = Polygon(label=label, points=points, conf=confidence)
                            is_object = True
                        if bb is not None:
                            x_min, y_min, w_box, h_box = one_obj['bbox']
                            x_min = int(x_min)
                            y_min = int(y_min)
                            x_max = int(x_min) + int(w_box)
                            y_max = int(y_min) + int(h_box)
                            box = HBox(label=label, x_min=x_min, y_min=y_min,
                                       x_max=x_max, y_max=y_max, conf=confidence)
                            is_object = True
                        if kp is not None:
                            points = [(kp[i], kp[i + 1], kp[i + 2]) for i in
                                      range(0, len(kp), 3)]
                            keypoints = Keypoints(label=label, points=points, point_names=kp_names, skeleton=sk,
                                                  conf=confidence)
                            is_object = True

                        if is_object:
                            obj = Object(hbox=box, keypoints=keypoints, polygon=polygon)
                            ann.add(obj=obj)

                            for shape, v in obj:
                                if label in self._labels_stat[shape]:
                                    self._labels_stat[shape][label] += 1
                                else:
                                    self._labels_stat[shape][label] = 1

            if len(ann) > 0 or is_load_empty_ann:
                self._annotations += [ann]

        self.log.info('Loaded {} {} annotations.'.format(
            len(self.annotations), self.__class__.__name__))

    def save(self, save_dir: str = None, is_save_images: bool = False, **kwargs):
        """
        Save annotations to MSCOCO format
        :param save_dir: save directory path. If save_dir is None, write annotations to current directory
        :param is_save_images: if True - save images, otherwise - do not save image files
        :return:
        """
        assert len(self.annotations) > 0

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            time_now = datetime.datetime.strftime(
                datetime.datetime.today(), '%m-%d-%H-%M')
            target_dir = os.path.join(
                save_dir, 'annotations_MSCOCO_' + time_now)
            os.makedirs(target_dir, exist_ok=True)
        else:
            if self.annotation_path is None:
                raise ValueError('Save directory is None.')
            target_dir = self.annotation_path

        if is_save_images:
            img_save_dir = os.path.join(os.path.dirname(target_dir), 'images')
            os.makedirs(img_save_dir, exist_ok=True)

        classes_dict = {}

        for i in range(len(self.labels)):
            classes_dict[self.labels[i]] = i + 1

        # creating fields for an annotation file in COCO format
        dataset_json = OrderedDict()
        dataset_json['images'] = []
        dataset_json['type'] = 'instances'
        dataset_json['annotations'] = []
        dataset_json['categories'] = []

        index_for_annotations = 1
        cur_categories_names = []
        dataset_categories = []
        for index, ann in tqdm(enumerate(self.annotations)):
            img_path = os.path.basename(ann.img_path)
            width = ann.width
            height = ann.height
            id_img = index + 1

            one_img = {'id': id_img, 'file_name': img_path,
                       'width': width, 'height': height}
            dataset_json['images'].append(one_img)

            for obj_shape in ann.objects:
                one_ann = dict()
                one_ann['id'] = index_for_annotations
                one_ann['image_id'] = id_img
                one_ann['category_id'] = classes_dict[obj_shape.label]
                one_ann['iscrowd'] = 0
                one_ann['ignore'] = 0
                if obj_shape.keypoints is not None:
                    ann_points = self._prepare_one_obj(obj_shape.keypoints)
                    one_ann.update(ann_points)
                if obj_shape.polygon is not None:
                    ann_points = self._prepare_one_obj(obj_shape.polygon)
                    one_ann.update(ann_points)
                if obj_shape.hbox is not None:
                    ann_points = self._prepare_one_obj(obj_shape.hbox)
                    one_ann.update(ann_points)

                dataset_json['annotations'].append(one_ann)
                index_for_annotations += 1

                if obj_shape.label not in cur_categories_names:
                    cur_categories_names.append(obj_shape.label)
                    one_category = dict()
                    one_category['id'] = classes_dict[obj_shape.label]
                    one_category['name'] = obj_shape.label
                    one_category['supercategory'] = ''
                    if obj_shape.keypoints is not None:
                        one_category['keypoints'] = obj_shape.keypoints.point_names
                        one_category['skeleton'] = obj_shape.keypoints.skeleton

                    dataset_categories.append(one_category)

        dataset_json['categories'] = dataset_categories

        target_dir = os.path.join(target_dir, 'annotations_MSCOCO.json')
        with open(target_dir, 'w', encoding='utf8') as f:
            json.dump(dataset_json, f)

        self.log.info("{}. Saved {} annotations.".format(
            self.__class__.__name__, len(self.annotations)))

        if is_save_images:
            self.log.info("{}. Saved {} images.".format(
                self.__class__.__name__, len(self.annotations)))

        self.log.info("Saved in {}.".format(self.__class__.__name__))

    def _prepare_one_obj(self, ann_obj: Union[Polygon, HBox]):
        """
        Prepare one annotation object to save
        :param ann_obj: Polygon or HBox object
        :return:
        """
        new_obj = dict()

        if isinstance(ann_obj, Polygon):
            new_obj['area'] = ann_obj.area
            new_obj['segmentation'] = [[coord for point in ann_obj.points for coord in point]]
            bbox_obj = HBox.from_polygon(ann_obj)
            bbox = bbox_obj.box
            new_obj['bbox'] = [bbox[0], bbox[1], round((bbox[2] - bbox[0]), 2), round((bbox[3] - bbox[1]), 2)]
            new_obj['score'] = ann_obj.confidence
        elif isinstance(ann_obj, HBox):
            bbox = ann_obj.box
            new_obj['area'] = ann_obj.pixel_area
            new_obj['bbox'] = [bbox[0], bbox[1], round((bbox[2] - bbox[0]), 2), round((bbox[3] - bbox[1]), 2)]
            new_obj['score'] = ann_obj.confidence
        elif isinstance(ann_obj, Keypoints):
            new_obj['keypoints'] = [coord for point in ann_obj.points for coord in point]
            new_obj['num_keypoints'] = len(ann_obj.points)

        return new_obj
