import logging
import os

logger = logging.getLogger(os.path.basename(os.path.dirname(__file__)))
logger.setLevel(logging.INFO)

# stream logging
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

from .__version__ import __version__, __author__, __url__, __license__
from .shapes.hbox import HBox
from .shapes.polygon import Polygon
from .formats.pascalvoc import PascalVOC
from .formats.yolo import Yolo
from .formats.yolov4 import Yolov4
from .formats.labelme import LabelMe
from .formats.mscoco import MSCOCO
from .formats.base import FormatBase
from .formats.via import VIA

