import os
import datetime
import shutil
from tqdm import tqdm
import xmltodict
import xml.etree.ElementTree as ET
from typing import List
from .base import FormatBase
from ..shapes.polygon import Polygon
from ..shapes.hbox import HBox
from ..annotation import Annotation
from ..utils import Image, colorstr


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
        super()._load(labels=labels, is_load_empty_ann=is_load_empty_ann)

        for img_name in tqdm(os.listdir(self._img_dir)):
            image_path = os.path.join(self._img_dir, img_name)
            ann_path = os.path.join(self._ann_dir, os.path.splitext(img_name)[0] + '.xml')
            width, height, depth = Image.get_shape(img_path=image_path)
            cur_ann = Annotation(img_path=image_path, width=width, height=height, depth=depth)

            try:
                xml = xmltodict.parse(open(ann_path, 'rb'))
                # if now objects
                if 'object' in xml['annotation']:
                    # if one object in xml
                    if 'name' in xml['annotation']['object']:
                        objects = [xml['annotation']['object']]
                    else:
                        objects = xml['annotation']['object']
                    for obj in objects:
                        points = []
                        for p in obj['polygon']['pt']:
                            points.append((float(p['x']), float(p['y'])))

                        polygon = Polygon(label=obj['name'], points=points)
                        cur_ann.add(polygon)

                if cur_ann is not None and (len(cur_ann) > 0 or is_load_empty_ann):
                    self._annotations += [cur_ann]
            except Exception as error:
                self.log.info('{}: Skip annotation saving: {}'.format(colorstr(self.__class__.__name__), error))
                continue

        self.log.info('Loaded {} {} annotations.'.format(colorstr('cyan', 'bold', len(self.annotations)),
                                                         colorstr(self.__class__.__name__)))

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
            if self.annotation_directory is None:
                raise ValueError('Save directory is None.')
            target_dir = self.annotation_directory

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

            for idx, obj in enumerate(ann.objects):
                if isinstance(obj, HBox):
                    polygon = Polygon.from_box(obj)
                elif isinstance(obj, Polygon):
                    polygon = obj
                else:
                    raise NotImplemented

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
                obj_id.text = str(idx)

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
