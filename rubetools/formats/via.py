import json
import os
from typing import List
from tqdm import tqdm

from ..annotation import Annotation
from ..shapes.polygon import Polygon
from .base import FormatBase
from ..utils import Image, colorstr


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
        super()._load(labels=labels, is_load_empty_ann=is_load_empty_ann)

        with open(self._ann_dir, "r") as f_ann:
            dataset = json.load(f_ann)

        images_metadata = dataset["_via_img_metadata"]

        for _, img_data in tqdm(images_metadata.items()):
            img_name = img_data["filename"]
            img_path = os.path.join(self._img_dir, img_name)
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
