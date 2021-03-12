import os
from glob import glob
import shutil
from tqdm import tqdm
import datetime
import xml.etree.ElementTree as ET
import xmltodict
from typing import List, Union

from .base import FormatBase
from ..shapes.hbox import HBox
from ..shapes.polygon import Polygon
from ..annotation import Annotation
from ..utils import Image, colorstr


class PascalVOC(FormatBase):
    """
    PascalVOC annotation format
    """
    def _load(self, labels: List[str] = None, is_load_empty_ann: bool = True):
        """
        Load PascalVOC format annotations
        :param labels: list of class labels. If None, load all seen labels
        :param is_load_empty_ann: if True - load annotations with empty objects images, otherwise - skip.
        :return:
        """
        # common checks implemented in the base method
        super()._load(labels=labels, is_load_empty_ann=is_load_empty_ann)

        if not (os.path.isdir(self._img_dir) and os.path.isdir(self._ann_dir)):
            raise FileNotFoundError('Input paths are not directories.')

        img_paths = [f for f in glob(self._img_dir + '/*', recursive=True) if os.path.isfile(f)]
        for img_path in tqdm(img_paths):
            img_name = os.path.basename(img_path)
            ann_path = os.path.join(self._ann_dir, os.path.splitext(img_name)[0] + '.xml')

            ann = self.get_annotation(ann_path=ann_path, img_path=img_path, labels=labels)
            if ann is not None and (len(ann) > 0 or is_load_empty_ann):
                self._annotations += [ann]

        self.log.info('Loaded {} {} annotations.'.format(colorstr('cyan', 'bold', len(self.annotations)),
                                                         colorstr(self.__class__.__name__)))

    def get_annotation(self, ann_path: str, img_path: str = None, labels: List[str] = None) -> Union[Annotation, None]:
        """
        Load PascalVOC annotation object
        :param ann_path: annotation path
        :param img_path: image path
        :param labels: list of class labels. If None, load all seen labels
        :return:
        """
        try:
            with open(ann_path, 'rb') as f:
                ann_xml = xmltodict.parse(f)
        except Exception as error:
            self.log.info('{}: Skip annotation loading: {}'.format(colorstr(self.__class__.__name__), error))
            return None

        try:
            ann_xml = ann_xml['annotation']
            if img_path is not None and os.path.exists(img_path):
                width, height, depth = Image.get_shape(img_path)
            else:
                img_path = ann_xml['filename']
                width = int(ann_xml['size']['width'])
                height = int(ann_xml['size']['height'])
                if 'depth' in ann_xml['size'] and ann_xml['size']['depth'] is not None:
                    depth = int(ann_xml['size']['depth'])
                else:
                    depth = 3

            ann = Annotation(img_path=img_path, width=width, height=height, depth=depth)
            if 'object' in ann_xml:
                # if only one object in annotation
                boxes = ann_xml['object'] if 'name' not in ann_xml['object'] else [ann_xml['object']]
                for obj in boxes:
                    try:
                        obj_name = obj['name']
                        if labels is not None and len(labels) > 0 and obj_name not in labels:
                            continue
                        if obj_name in self._labels_stat:
                            self._labels_stat[obj_name] += 1
                        else:
                            self._labels_stat[obj_name] = 1

                        x_min = int(round(float(obj['bndbox']['xmin'])))
                        y_min = int(round(float(obj['bndbox']['ymin'])))
                        x_max = int(round(float(obj['bndbox']['xmax'])))
                        y_max = int(round(float(obj['bndbox']['ymax'])))

                        box = HBox(label=obj_name, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                        ann.add(box)
                    except KeyError:
                        self.log.info("{}: One of annotation param is not set.".format(colorstr(self.__class__.__name__)))
            return ann
        except KeyError as error:
            self.log.info('{}: Skip annotation loading: {}'.format(colorstr(self.__class__.__name__), error))
            return None

    def save(self, save_dir: str = None, is_save_images: bool = False, **kwargs):
        """
        Save annotations to PascalVOC format
        :param save_dir: save directory path. If save_dir is None, write annotations to current directory
        :param is_save_images: if True - save images, otherwise - do not save image files
        :return:
        """
        assert len(self.annotations) > 0

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            time_now = datetime.datetime.strftime(datetime.datetime.today(), '%m-%d-%H-%M')
            target_dir = os.path.join(save_dir, 'annotations_pascal_' + time_now)
            os.makedirs(target_dir, exist_ok=True)
        else:
            if self.annotation_directory is None:
                raise ValueError('Save directory is None.')
            target_dir = self.annotation_directory

        if is_save_images:
            img_save_dir = os.path.join(os.path.dirname(target_dir), 'images')
            os.makedirs(img_save_dir, exist_ok=True)

        saved_img_counter = 0
        for ann in tqdm(self.annotations):
            root = ET.Element('annotation')
            folder = ET.SubElement(root, 'folder')
            folder.text = os.path.basename(os.path.dirname(ann.img_path))

            filename = ET.SubElement(root, 'filename')
            filename.text = os.path.basename(ann.img_path)

            path = ET.SubElement(root, 'path')
            path.text = ann.img_path

            size = ET.SubElement(root, 'size')
            width = ET.SubElement(size, 'width')
            height = ET.SubElement(size, 'height')
            depth = ET.SubElement(size, 'depth')

            width.text = str(ann.width)
            height.text = str(ann.height)
            depth.text = str(ann.depth)

            for obj in ann.objects:
                if isinstance(obj, HBox):
                    hbox = obj
                elif isinstance(obj, Polygon):
                    hbox = HBox.from_polygon(obj)
                else:
                    raise NotImplemented

                box = ET.SubElement(root, 'object')

                name = ET.SubElement(box, 'name')
                name.text = hbox.label

                bndbox = ET.SubElement(box, 'bndbox')
                xmin = hbox.box[0]
                ymin = hbox.box[1]
                xmax = hbox.box[2]
                ymax = hbox.box[3]

                bndbox_xmin = ET.SubElement(bndbox, 'xmin')
                bndbox_xmin.text = str(xmin)
                bndbox_ymin = ET.SubElement(bndbox, 'ymin')
                bndbox_ymin.text = str(ymin)
                bndbox_xmax = ET.SubElement(bndbox, 'xmax')
                bndbox_xmax.text = str(xmax)
                bndbox_ymax = ET.SubElement(bndbox, 'ymax')
                bndbox_ymax.text = str(ymax)

            tree = ET.ElementTree(root)
            xml_path = os.path.join(target_dir, os.path.splitext(os.path.basename(ann.img_path))[0] + '.xml')
            tree.write(xml_path)

            if is_save_images and os.path.exists(ann.img_path):
                new_path = os.path.join(img_save_dir, os.path.basename(ann.img_path))
                shutil.copy(ann.img_path, new_path)
                saved_img_counter += 1

        self.log.info("{}: Saved {} annotations.".format(colorstr(self.__class__.__name__),
                                                         colorstr('cyan', 'bold', len(self.annotations))))

        if is_save_images:
            self.log.info("{}: Saved {} images.".format(colorstr(self.__class__.__name__),
                                                        colorstr('cyan', 'bold', saved_img_counter)))

        self.log.info("Saved in {}.".format(colorstr(self.__class__.__name__)))
