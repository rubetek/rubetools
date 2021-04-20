import os
import datetime
import shutil
from glob import glob
from tqdm import tqdm
import xmltodict
import xml.etree.ElementTree as ET
from typing import List, Union
from .base import FormatBase
from ..shapes.polygon import Polygon
from ..shapes.hbox import HBox
from ..annotation import Annotation
from ..utils import Image


class LabelMe(FormatBase):
    """
    LabelMe annotation format
    """

    def _load(self, labels: List[str] = None, is_load_empty_ann: bool = True):
        """
        Load LabelMe format annotations
        :param labels: list of class labels. If None, load all seen labels
        :param is_load_empty_ann: if True - load annotations with empty objects images, otherwise - skip.
        :return:
        """
        if self._img_path is not None and isinstance(self._img_path, str) and os.path.isdir(self._img_path):
            paths = [f for f in glob(self._img_path + '/*', recursive=True) if os.path.isfile(f)]
        elif self._ann_path is not None and isinstance(self._ann_path, str) and os.path.isdir(self._ann_path):
            paths = [f for f in glob(self._ann_path + '/*', recursive=True) if os.path.isfile(f)]
        else:
            raise FileNotFoundError('Incorrect initialization paths.')

        for p in tqdm(paths):
            if p.split('.')[-1] != 'xml':
                img_name = os.path.basename(p)
                ann_path = os.path.join(self._ann_path, os.path.splitext(img_name)[0] + '.xml')
                ann = self.get_annotation(ann_path=ann_path, img_path=p, labels=labels)
            else:
                ann = self.get_annotation(ann_path=p, img_path=None, labels=labels)

            if ann is not None and (len(ann) > 0 or is_load_empty_ann):
                self._annotations += [ann]

        self.log.info('Loaded {} {} annotations.'.format(len(self.annotations), self.__class__.__name__))

    def get_annotation(self, ann_path: str, img_path: str = None, labels: List[str] = None) -> Union[Annotation, None]:
        """
        Load LabelMe annotation object
        :param ann_path: annotation path
        :param img_path: image path
        :param labels: list of class labels. If None, load all seen labels
        :return:
        """
        try:
            with open(ann_path, 'rb') as f:
                ann_xml = xmltodict.parse(f)
        except Exception as error:
            self.log.info('{}. Skip annotation loading: {}'.format(self.__class__.__name__, error))
            return None

        try:
            ann_xml = ann_xml['annotation']
            if img_path is not None and os.path.exists(img_path):
                width, height, depth = Image.get_shape(img_path)
            else:
                if 'folder' in ann_xml and ann_xml['folder'] is not None:
                    img_path = os.path.join(ann_xml['folder'], ann_xml['filename'])
                else:
                    img_path = ann_xml['filename']
                width = int(ann_xml['imagesize']['ncols'])
                height = int(ann_xml['imagesize']['nrows'])
                depth = 3

            ann = Annotation(img_path=img_path, width=width, height=height, depth=depth)
            if 'object' in ann_xml:
                # if one object in xml
                if 'name' in ann_xml['object']:
                    objects = [ann_xml['object']]
                else:
                    objects = ann_xml['object']
                for obj in objects:
                    if labels is not None and len(labels) > 0 and obj['name'] not in labels:
                        continue
                    if obj['name'] in self._labels_stat['polygon']:
                        self._labels_stat['polygon'][obj['name']] += 1
                    else:
                        self._labels_stat['polygon'][obj['name']] = 1

                    points = []
                    for p in obj['polygon']['pt']:
                        points.append((float(p['x']), float(p['y'])))

                    polygon = Polygon(label=obj['name'], points=points)
                    ann.add(polygon)
            return ann
        except Exception as error:
            self.log.info('{}. Skip annotation loading: {}'.format(self.__class__.__name__, error))
            return None

    def save(self, save_dir: str = None, is_save_images: bool = False, **kwargs):
        """
        Save annotations to LabelMe format
        :param save_dir: save directory path. If save_dir is None, write annotations to current directory
        :param is_save_images: if True - save images, otherwise - do not save image files
        :return:
        """
        assert len(self.annotations) > 0

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            time_now = datetime.datetime.strftime(datetime.datetime.today(), '%m-%d-%H-%M')
            target_dir = os.path.join(save_dir, 'annotations_labelme_' + time_now)
            os.makedirs(target_dir, exist_ok=True)
        else:
            if self.annotation_path is None:
                raise ValueError('Save directory is None.')
            target_dir = self.annotation_path

        if is_save_images:
            img_save_dir = os.path.join(os.path.dirname(target_dir), 'images')
            os.makedirs(img_save_dir, exist_ok=True)

        saved_img_counter = 0
        for ann in tqdm(self.annotations):
            root = ET.Element('annotation')
            filename = ET.SubElement(root, 'filename')
            filename.text = os.path.basename(ann.img_path)
            ET.SubElement(root, 'folder')

            source = ET.SubElement(root, 'source')
            ET.SubElement(source, 'sourceImage')
            source_ann = ET.SubElement(source, 'sourceAnnotation')
            source_ann.text = 'rubetools'

            size = ET.SubElement(root, 'imagesize')
            height = ET.SubElement(size, 'nrows')
            height.text = str(ann.height)
            width = ET.SubElement(size, 'ncols')
            width.text = str(ann.width)

            obj_counter = 0
            for obj_shape in ann.objects:
                for _, obj in obj_shape:
                    if isinstance(obj, HBox):
                        polygon = Polygon.from_box(obj)
                    elif isinstance(obj, Polygon):
                        polygon = obj
                    else:
                        continue

                    box = ET.SubElement(root, 'object')

                    name = ET.SubElement(box, 'name')
                    name.text = polygon.label

                    deleted = ET.SubElement(box, 'deleted')
                    deleted.text = '0'
                    verified = ET.SubElement(box, 'verified')
                    verified.text = '0'
                    occluded = ET.SubElement(box, 'occluded')
                    occluded.text = 'no'
                    date = ET.SubElement(box, 'date')
                    date.text = str(datetime.datetime.now())
                    obj_id = ET.SubElement(box, 'id')
                    obj_id.text = str(obj_counter)

                    parts = ET.SubElement(box, 'parts')
                    ET.SubElement(parts, 'hasparts')
                    ET.SubElement(parts, 'ispartof')

                    obj_polygon = ET.SubElement(box, 'polygon')
                    for p in polygon.points:
                        pt = ET.SubElement(obj_polygon, 'pt')
                        pt_x = ET.SubElement(pt, 'x')
                        pt_x.text = str(p[0])
                        pt_y = ET.SubElement(pt, 'y')
                        pt_y.text = str(p[1])

                    ET.SubElement(box, 'attributes')
                    obj_counter += 1

            tree = ET.ElementTree(root)
            xml_path = os.path.join(target_dir, os.path.splitext(os.path.basename(ann.img_path))[0] + '.xml')
            tree.write(xml_path)

            if is_save_images and os.path.exists(ann.img_path):
                new_path = os.path.join(img_save_dir, os.path.basename(ann.img_path))
                shutil.copy(ann.img_path, new_path)
                saved_img_counter += 1

        self.log.info("{}. Saved {} annotations.".format(self.__class__.__name__, len(self.annotations)))

        if is_save_images:
            self.log.info("{}. Saved {} images.".format(self.__class__.__name__, saved_img_counter))

        self.log.info("Saved in {}.".format(self.__class__.__name__))
