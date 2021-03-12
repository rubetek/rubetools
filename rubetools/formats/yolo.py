import os
import shutil
from tqdm import tqdm
import datetime
from typing import List, Union, Dict
from .base import FormatBase
from ..shapes.hbox import HBox
from ..shapes.polygon import Polygon
from ..annotation import Annotation
from ..utils import Image, colorstr


class Yolo(FormatBase):
    """
    Yolo annotation format
    """
    def __init__(self, ann_path: str = None, img_path: str = None, selected_labels: List[str] = None,
                 is_load_empty_ann: bool = True, file_cls_name: str = 'classes.txt'):
        self._file_cls_name = file_cls_name
        super().__init__(ann_path=ann_path, img_path=img_path, selected_labels=selected_labels,
                         is_load_empty_ann=is_load_empty_ann)

    def _load(self, labels: List[str] = None, is_load_empty_ann: bool = True):
        """
        Load Yolo format annotations
        :param labels: list of class labels. If None, load all seen labels
        :param is_load_empty_ann: if True - load annotations with empty objects images, otherwise - skip.
        :return:
        """
        # common checks implemented in the base method
        super()._load(labels=labels, is_load_empty_ann=is_load_empty_ann)

        if not (os.path.isdir(self._img_dir) and os.path.isdir(self._ann_dir)):
            raise FileNotFoundError('Input paths are not directories.')

        # dict for pairs (cls_id, cls_name)
        class_dict = {}
        with open(os.path.join(self._ann_dir, self._file_cls_name), 'r') as f_cls:
            classes = f_cls.read().splitlines()
        for cls_id, cls_name in enumerate(classes):
            class_dict[cls_id] = cls_name

        # create annotations
        for img_name in tqdm(os.listdir(self._img_dir)):
            img_path = os.path.join(self._img_dir, img_name)
            ann_path = os.path.join(self._ann_dir, os.path.splitext(img_name)[0] + '.txt')

            ann = self.get_annotation(ann_path=ann_path, img_path=img_path, class_dict=class_dict, labels=labels)
            if ann is not None and (len(ann) > 0 or is_load_empty_ann):
                self._annotations += [ann]

        self.log.info('Loaded {} {} annotations.'.format(colorstr('cyan', 'bold',len(self.annotations)),
                                                         colorstr(self.__class__.__name__)))

    def get_annotation(self, ann_path: str, img_path: str, class_dict: Dict[int, str],
                       labels: List[str] = None) -> Union[Annotation, None]:
        """
        Load Yolo annotation object
        :param ann_path: annotation path
        :param img_path: image path
        :param class_dict: ordered label dictionary
        :param labels: list of class labels. If None, load all seen labels
        :return:
        """
        try:
            with open(ann_path, 'r') as ann_file:
                all_annotations = ann_file.read().splitlines()
        except Exception as error:
            self.log.info('{}. Skip annotation loading: {}'.format(colorstr(self.__class__.__name__), error))
            return None

        if img_path is None or not os.path.exists(img_path):
            self.log.info('{}: Skip annotation loading: Image is not exists.'.format(colorstr(self.__class__.__name__)))
            return None

        w, h, d = Image.get_shape(img_path=img_path)
        ann = Annotation(img_path=img_path, width=w, height=h, depth=d)

        for one_obj in all_annotations:
            one_obj = one_obj.split(' ')
            obj_name = class_dict[int(one_obj[0])]
            if len(labels) > 0 and obj_name not in labels:
                continue
            if obj_name in self._labels_stat:
                self._labels_stat[obj_name] += 1
            else:
                self._labels_stat[obj_name] = 1

            x_c = float(one_obj[1])
            y_c = float(one_obj[2])
            w_box = float(one_obj[3])
            h_box = float(one_obj[4])

            x_min = int((x_c * w) - (w_box * w) / 2 + 0.5)
            y_max = int((y_c * h) + (h_box * h) / 2 + 0.5)
            x_max = int((x_c * w) + (w_box * w) / 2 + 0.5)
            y_min = int((y_c * h) - (h_box * h) / 2 + 0.5)

            box = HBox(label=obj_name, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
            ann.add(box)

        return ann

    def save(self, save_dir: str = None, is_save_images: bool = False, **kwargs):
        """
        Save annotations to Yolo format
        :param save_dir: save directory path. If save_dir is None, write annotations to current directory
        :param is_save_images: if True - save images, otherwise - do not save image files
        :return:
        """
        assert len(self.annotations) > 0

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            time_now = datetime.datetime.strftime(datetime.datetime.today(), '%m-%d-%H-%M')
            target_dir = os.path.join(save_dir, 'annotations_yolo_' + time_now)
            os.makedirs(target_dir, exist_ok=True)
        else:
            if self.annotation_directory is None:
                raise ValueError('Save directory is None.')
            target_dir = self.annotation_directory

        # upload dir classes name
        classes_dir = {}
        name_classes = list(self._labels_stat.keys())

        for i in range(len(name_classes)):
            classes_dir[name_classes[i]] = i

        if is_save_images:
            img_save_dir = os.path.join(os.path.dirname(target_dir), 'images')
            os.makedirs(img_save_dir, exist_ok=True)

        for ann in tqdm(self.annotations):
            path_ann = os.path.join(target_dir, os.path.splitext(os.path.basename(ann.img_path))[0] + '.txt')
            with open(path_ann, 'w') as f:
                w_image = ann.width
                h_image = ann.height

                for obj in ann.objects:
                    if isinstance(obj, HBox):
                        hbox = obj
                    elif isinstance(obj, Polygon):
                        hbox = HBox.from_polygon(obj)
                    else:
                        raise NotImplemented

                    label = classes_dir[hbox.label]

                    x_c = float((hbox.box[0] + hbox.box[2])/2) / w_image
                    y_c = float((hbox.box[1] + hbox.box[3])/2) / h_image
                    w_b = float((hbox.box[2] - hbox.box[0]) / w_image)
                    h_b = float((hbox.box[3] - hbox.box[1]) / h_image)

                    box = '%d %f %f %f %f\r' % (label, x_c, y_c, w_b, h_b)
                    f.writelines(box)

            if is_save_images:
                new_path = os.path.join(img_save_dir, os.path.basename(ann.img_path))
                shutil.copy(ann.img_path, new_path)

        file_classes_path = os.path.join(target_dir, self._file_cls_name)
        with open(file_classes_path, 'w') as f:
            for cls_name, id_cls in classes_dir.items():
                str_clas_and_id = '{}\r'.format(cls_name)
                f.writelines(str_clas_and_id)

        self.log.info("{}: Saved {} annotations.".format(colorstr(self.__class__.__name__),
                                                         colorstr('cyan', 'bold', len(self.annotations))))

        if is_save_images:
            self.log.info("{}: Saved {} images.".format(colorstr(self.__class__.__name__),
                                                        colorstr('cyan', 'bold', len(self.annotations))))

        self.log.info("Saved in {}.".format(colorstr(self.__class__.__name__)))

    def save_paths_list(self, new_images_dir: str, dest_dir: str):
        """
        Renaming the base directory in the file with the list of paths to images
        :param new_images_dir:
        :param dest_dir: destination directory for new file
        :return:
        """
        new_img_file_path = os.path.join(dest_dir, 'images_path.txt')
        with open(new_img_file_path, 'w') as f:
            for ann in tqdm(self.annotations):
                old_img_path = ann.img_path
                img_name = os.path.basename(old_img_path)
                new_img_path = os.path.join(new_images_dir, img_name)
                str_image_path = '{}\r'.format(new_img_path)
                f.writelines(str_image_path)
