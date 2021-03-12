import os
from tqdm import tqdm
import datetime
from collections import OrderedDict
import json
import shutil
from typing import List, Union
from .base import FormatBase
from ..shapes.hbox import HBox
from ..shapes.polygon import Polygon
from ..annotation import Annotation
from ..utils import Image, colorstr


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
                "ignore":
            },
            ...
        ],
        "categories": [
            {
                "supercategory": ,
                "id": ,
                "name":
            },
            ...
        ]
    }
    """

    def _load(self, labels: List[str] = None, is_load_empty_ann: bool = True):
        """
        Load MSCOCO format annotations
        :param labels: list of class labels. If None, load all seen labels
        :param is_load_empty_ann: if True - load annotations with empty objects images, otherwise - skip.
        :return:
        """
        # common checks implemented in the base method
        super()._load(labels=labels, is_load_empty_ann=is_load_empty_ann)

        with open(self._ann_dir, 'r') as f_ann:
            dataset = json.load(f_ann)

        categories = dataset['categories']
        images = dataset['images']
        annotations = dataset['annotations']

        # creating a dictionary of object categories in a dataset: {id_class: name_class}
        classes_dict = dict()
        for class_obj in categories:
            classes_dict[class_obj['id']] = class_obj['name']

        for img in tqdm(images):
            img_name = img['file_name']
            img_path = os.path.join(self._img_dir, img_name)
            width, height, depth = Image.get_shape(img_path=img_path)
            img_id = img['id']

            ann = Annotation(img_path=img_path, width=width, height=height, depth=depth)
            for one_obj in annotations:
                # check if the next object belongs to our image
                if one_obj['image_id'] == img_id:
                    label = classes_dict[one_obj["category_id"]]

                    if label in self._labels_stat:
                        self._labels_stat[label] += 1
                    else:
                        self._labels_stat[label] = 1

                    if len(labels) > 0 and label not in labels:
                        continue
                    else:
                        x_min, y_min, w_box, h_box = one_obj['bbox']

                        x_min = int(x_min)
                        y_min = int(y_min)
                        x_max = int(x_min) + int(w_box)
                        y_max = int(y_min) + int(h_box)

                        box = HBox(label=label, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                        ann.add(box)

            if len(ann) > 0 or is_load_empty_ann:
                self._annotations += [ann]

        self.log.info('Loaded {} {} annotations.'.format(colorstr('cyan', 'bold', len(self.annotations)),
                                                         colorstr(self.__class__.__name__)))

    def _prepare_one_obj(self, ann_obj: Union[Polygon, HBox]):
        """
        Prepare one annotation object to save
        :param ann_obj: Polygon or HBox object
        :return:
        """
        new_obj = dict()

        if isinstance(ann_obj, Polygon):
            new_obj['area'] = ann_obj.area
            new_obj['segmentation'] = [coord for point in ann_obj.points for coord in point]
            bbox_obj = HBox.from_polygon(ann_obj)
            bbox = bbox_obj.box
        elif isinstance(ann_obj, HBox):
            bbox = ann_obj.box
            new_obj['area'] = round((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1), 2)
            new_obj['segmentation'] = []

        bbox_coco = [bbox[0], bbox[1], round((bbox[2] - bbox[0]), 2), round((bbox[3] - bbox[1]), 2)]
        new_obj['bbox'] = bbox_coco

        return new_obj

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
            time_now = datetime.datetime.strftime(datetime.datetime.today(), '%m-%d-%H-%M')
            target_dir = os.path.join(save_dir, 'annotations_MSCOCO_' + time_now)
            os.makedirs(target_dir, exist_ok=True)
        else:
            if self.annotation_directory is None:
                raise ValueError('Save directory is None.')
            target_dir = self.annotation_directory

        if is_save_images:
            img_save_dir = os.path.join(os.path.dirname(target_dir), 'images')
            os.makedirs(img_save_dir, exist_ok=True)

        # creating fields for an annotation file in COCO format
        dataset_json = OrderedDict()
        dataset_json['images'] = []
        dataset_json['type'] = 'instances'
        dataset_json['annotations'] = []
        dataset_json['categories'] = []
        
        # creating a dictionary {name_class: id_class}
        classes_dict = {}
        name_classes = list(self._labels_stat.keys())

        for i in range(len(name_classes)):
            classes_dict[name_classes[i]] = i + 1

        # creating the 'categories' field for the annotation file in COCO format
        for name_class, id_class in classes_dict.items():
            one_category = dict()
            one_category['id'] = id_class
            one_category['name'] = name_class
            one_category['supercategory'] = ''

            dataset_json['categories'].append(one_category)

        # creating an 'images' field and a '_annotations' field for an annotation file in COCO format
        index_for_annotations = 1
        for index, ann in tqdm(enumerate(self.annotations)):
            img_path = os.path.basename(ann.img_path)
            width = ann.width
            height = ann.height
            id_img = index + 1

            one_img = {'id': id_img, 'file_name': img_path, 'width': width, 'height': height}
            dataset_json['images'].append(one_img)

            for obj in ann.objects:
                one_ann = dict()
                one_ann['id'] = index_for_annotations
                one_ann['image_id'] = id_img
                one_ann['category_id'] = classes_dict[obj.label]
                one_ann['iscrowd'] = 0
                one_ann['ignore'] = 0

                ann_points = self._prepare_one_obj(obj)
                one_ann.update(ann_points)

                dataset_json['annotations'].append(one_ann)
                index_for_annotations += 1

            if is_save_images:
                new_path = os.path.join(img_save_dir, os.path.basename(ann.img_path))
                shutil.copy(ann.img_path, new_path)

        target_dir = os.path.join(target_dir, 'annotations_MSCOCO.json')
        with open(target_dir, 'w') as f:
            json.dump(dataset_json, f)

        self.log.info("{}: Saved {} annotations.".format(colorstr(self.__class__.__name__),
                                                         colorstr('cyan', 'bold', len(self.annotations))))

        if is_save_images:
            self.log.info("{}: Saved {} images.".format(colorstr(self.__class__.__name__),
                                                        colorstr('cyan', 'bold', len(self.annotations))))

        self.log.info("Saved in {}.".format(colorstr(self.__class__.__name__)))
