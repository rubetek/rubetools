import numpy as np
import cv2
from PIL import Image as PilImage
from typing import Tuple


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


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

