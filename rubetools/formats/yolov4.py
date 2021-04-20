import os
import shutil
from tqdm import tqdm
from typing import List
from .base import FormatBase
from ..shapes.hbox import HBox
from ..shapes.polygon import Polygon


class Yolov4(FormatBase):
    """
    Yolo Darknet annotation format
    """

    def _load(self, labels: List[str] = None, is_load_empty_ann: bool = True):
        """
        Load Yolo Darknet format annotations
        :param labels: list of class labels. If None, load all seen labels
        :param is_load_empty_ann: if True - load annotations with empty objects images, otherwise - skip.
        :return:
        """
        raise NotImplemented

    def save(self, save_dir: str = None, is_save_images: bool = False, img_list_name: str = 'train.txt', **kwargs):
        """
        Save annotations to Yolo Darknet (Yolov4) format
        :param save_dir: save directory path. If save_dir is None, write annotations to current directory
        :param is_save_images: if True - save images, otherwise - do not save image files
        :param img_list_name: The name of the file
                              to which the paths to all images will be written: 'train.txt' or 'test.txt'
        :return:
        """
        assert len(self._annotations) > 0

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        else:
            if self.annotation_path is None:
                raise ValueError('Save directory is None.')

        obj_names_path = os.path.join(save_dir, 'obj.names')
        class_names = self.labels
        with open(obj_names_path, mode="w", encoding='utf-8') as f:
            for class_name in class_names:
                f.write(class_name + '\n')

        obj_dir = os.path.join(save_dir, 'obj')
        os.makedirs(obj_dir, exist_ok=True)
        for ann in tqdm(self._annotations):
            path_ann = os.path.join(obj_dir, os.path.splitext(os.path.basename(ann.img_path))[0] + '.txt')
            with open(path_ann, 'w') as f:
                w_image = ann.width
                h_image = ann.height

                for obj_shape in ann.objects:
                    for _, obj in obj_shape:
                        if isinstance(obj, HBox):
                            hbox = obj
                        elif isinstance(obj, Polygon):
                            hbox = HBox.from_polygon(obj)
                        else:
                            continue

                        label = class_names.index(hbox.label)

                        x_c = float((hbox.box[0] + hbox.box[2])/2) / w_image
                        y_c = float((hbox.box[1] + hbox.box[3])/2) / h_image
                        w_b = float((hbox.box[2] - hbox.box[0]) / w_image)
                        h_b = float((hbox.box[3] - hbox.box[1]) / h_image)

                        box = '%d %f %f %f %f\r' % (label, x_c, y_c, w_b, h_b)
                        f.writelines(box)

            new_path = os.path.join(obj_dir, os.path.basename(ann.img_path))
            shutil.copy(ann.img_path, new_path)

        img_list_path = os.path.join(save_dir, img_list_name)
        with open(img_list_path, mode="w", encoding='utf-8') as f:
            for ann in self._annotations:
                new_path = os.path.join('obj', os.path.basename(ann.img_path))
                f.write(new_path + '\n')

        obj_data_path = os.path.join(save_dir, 'obj.data')
        with open(obj_data_path, 'w', encoding='utf-8') as f:
            f.write('classes = {}\n'.format(len(class_names)))
            f.write('train = train.txt\n')
            f.write('valid = test.txt\n')
            f.write('names = obj.names\n')
            f.write('backup = backup\n')

        self.log.info("Saved in {}.".format(str(self.__class__.__name__)))
