import numpy as np
import cv2
from PIL import Image as PilImage
from typing import Tuple


class Image:
    @staticmethod
    def get_shape(img_path: str) -> Tuple[int, int, int]:
        """
        Return image shape list [width, height, depth] without loading image buffer
        :param img_path: image path
        :return:
        """
        img = PilImage.open(img_path)
        width, height = img.size
        mode_to_bpp = {'1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32}
        depth = mode_to_bpp[img.mode] // 8
        img.close()
        return width, height, depth

    @staticmethod
    def get_buffer(img_path: str) -> np.ndarray:
        """
        Load and return image buffer
        :param img_path: image path
        :return: numpy image buffer
        """
        return cv2.imread(img_path)

    @staticmethod
    def save_buffer(save_path: str, buffer):
        """
        Save image buffer to save_path
        :param save_path: save image path
        :param buffer: numpy image buffer
        :return:
        """
        cv2.imwrite(save_path, buffer)

