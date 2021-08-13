import json
import os
from typing import List
from tqdm import tqdm

from ..annotation import Annotation
from ..shapes.polygon import Polygon
from .base import FormatBase
from ..utils import Image


class VIA(FormatBase):
    """
    VGG Image Annotator annotation format
    """

    def _load(self, labels: List[str] = None, is_load_empty_ann: bool = True):
        """
        Load VGG Image Annotator format annotations
        :param labels: list of class labels. If None, load all seen labels
        :param is_load_empty_ann: if True - load annotations with empty objects images, otherwise - skip.
        :return:
        """
        if self._img_path is None or not isinstance(self._img_path, str) or not os.path.isdir(self._img_path):
            raise NotADirectoryError("{}. Skip annotation loading: Image dir is empty or not exists.\n "
                                     "Couldn't evaluate shape image params.".format(self.__class__.__name__))

        if self._ann_path is not None and isinstance(self._ann_path, str) and os.path.isfile(self._ann_path):
            with open(self._ann_path, "r") as f_ann:
                dataset = json.load(f_ann)
        else:
            raise FileNotFoundError('Incorrect annotation path.')

        """
         Previous VGG Image Annotator JSON format used the metadata key. 
         # images_metadata = dataset["_via_img_metadata"]
         Now it's updated to the image name
        """
        images_metadata = dataset

        for _, img_data in tqdm(images_metadata.items()):
            img_name = img_data["filename"]
            img_path = os.path.join(self._img_path, img_name)
            width, height, depth = Image.get_shape(img_path=img_path)

            ann = Annotation(img_path=img_path, width=width, height=height, depth=depth)

            for region in img_data["regions"]:
                shape_attributes = region["shape_attributes"]
                region_attributes = region["region_attributes"]

                if len(region_attributes.items()) > 0:
                    region_label = region_attributes["type"]
                elif not len(region_attributes.items()) and len(labels) == 1:
                    region_label = labels[0]
                else:
                    raise ValueError(
                        "No labels provided in annotation file and expected labels > 1"
                    )

                if labels is not None and len(labels) > 0 and region_label not in labels:
                    continue
                if region_label in self._labels_stat['polygon']:
                    self._labels_stat['polygon'][region_label] += 1
                else:
                    self._labels_stat['polygon'][region_label] = 1

                if shape_attributes["name"] == "polygon":
                    points = [
                        (float(x), float(y))
                        for x, y in zip(
                            shape_attributes["all_points_x"],
                            shape_attributes["all_points_y"],
                        )
                    ]
                    polygon = Polygon(label=region_label, points=points)
                    ann.add(polygon)

            self._annotations += [ann]

        self.log.info(
            "Loaded {} {} annotations.".format(
                len(self.annotations), self.__class__.__name__
            )
        )

    def save(self, save_dir: str = None, is_save_images: bool = False, **kwargs):
        """
        Save annotations to VGG Image Annotator format
        :param save_dir: save directory path. If save_dir is None, write annotations to current directory
        :param is_save_images: if True - save images, otherwise - do not save image files
        :return:
        """
        raise NotImplemented
