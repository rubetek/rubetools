import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import logging
from typing import List
import matplotlib.pyplot as plt

from .shapes.hbox import HBox
from .shapes.polygon import Polygon
from .annotation import Annotation


class EDA:
    """
    Annotations exploratory data analysis (EDA)
    """
    def __init__(self, annotations: List[Annotation]):
        self._annotations = annotations
        self.__info_df = None
        self.log = logging.getLogger(__name__)

    def __repr__(self):
        return '<{}, annotations: {}>'.format(str(self.__class__.__name__), len(self._annotations))

    @property
    def info_df(self):
        """
        Get annotations common info
        :return:
        """
        if self.__info_df is None:
            self.__info_df = self._get_dataset_info()
        return self.__info_df

    def _get_dataset_info(self) -> pd.DataFrame:
        """
        Return statistical information about dataset: sizes, shapes, class labels
        :return: statistic in Dataframe format
        """
        assert len(self._annotations) > 0

        columns = ['filename', 'img_width', 'img_height', 'box_width', 'box_height', 'label', 'type']
        info_list = []

        for ann in self._annotations:
            for obj in ann.objects:
                if isinstance(obj, HBox):
                    hbox = obj
                elif isinstance(obj, Polygon):
                    hbox = HBox.from_polygon(obj)
                else:
                    raise NotImplemented
                info_list.append([os.path.basename(ann.img_path), int(ann.width), int(ann.height),
                                  float(hbox.box[2] - hbox.box[0]), float(hbox.box[3] - hbox.box[1]),
                                  hbox.label, obj.__class__.__name__])

        info_df = pd.DataFrame(data=info_list, columns=columns)
        return info_df

    def plot_classes_histogram(self, w: int = 10, h: int = 7):
        """
        Plot class distribution histogram
        :param w: plot width
        :param h: plot height
        :return:
        """
        labels_series = self.info_df['label']
        classes_names = labels_series.unique()
        num_classes = len(classes_names)
        freq_classes = [sum(labels_series == class_name) for class_name in classes_names]

        # create plot
        fig, ax = plt.subplots(figsize=(w, h))

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
      
        plt.title("Class distribution", fontsize=20)
#         plt.xlabel("class name", color='gray', fontsize=14)
        plt.ylabel("frequency", color='gray', fontsize=14, fontweight='bold')

        rects = plt.bar(classes_names, freq_classes, color = 'lightskyblue')
        plt.xticks(np.arange(num_classes))
        ax.set_xticklabels(classes_names, rotation=45, fontdict={'horizontalalignment': 'center', 'size': 12})
         
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.show()
        # return num_classes, classes_names, freq_classes

    def plot_image_size_histogram(self, w: int = 10, h: int = 7):
        """
        Plot image size frequency distribution
        :param w: plot width
        :param h: plot height
        :return:
        """
        info_df = self.info_df[['img_width', 'img_height']]

        size_images = np.array(info_df)
        unique_size_images = np.unique(size_images, axis=0)

        if len(unique_size_images.shape) == 1:
            unique_size_images = [unique_size_images]

        freq_unique_size_images = [len(self.info_df[size_images == unique_size_image]['filename'].unique()) for
                                   unique_size_image in unique_size_images]

        # гистограмма распределения объектов на изображениях заданной размерности
        fig, ax = plt.subplots(figsize=(w, h))

        plt.title("Image size distribution", fontsize=20)
#         plt.xlabel("image size", color='gray', fontsize=14)
        plt.ylabel("frequency", color='gray', fontsize=14,fontweight='bold')

        rects = plt.bar(np.arange(len(unique_size_images)), freq_unique_size_images, color = 'lightskyblue')
        plt.xticks(np.arange(len(unique_size_images)))
        ax.set_xticklabels(unique_size_images, rotation=45, fontdict={'horizontalalignment': 'center', 'size': 12})

#         for i in range(len(unique_size_images)):
#             plt.text(i, freq_unique_size_images[i] + 0.5, freq_unique_size_images[i], horizontalalignment='center')
            
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.show()
        # return freq_unique_size_images, unique_size_images

    def plot_objects_frequency_scatter(self, class_name: str = None, w: int = 12, h: int = 8):
        """
        Plot image objects sizes frequency distribution
        :param class_name: monitoring only the specific class name
        :param w: plot width
        :param h: plot height
        :return:
        """
        if class_name:
            info_df = self.info_df[self.info_df['label'] == class_name]
        else:
            info_df = self.info_df

        box_widths = np.array(info_df['box_width'])
        box_heights = np.array(info_df['box_height'])

        # relative box shapes
        box_widths /= np.array(info_df['img_width'])
        box_heights /= np.array(info_df['img_height'])

        fig = plt.figure(figsize=(w, h))
        grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.5)

        main_ax = fig.add_subplot(grid[:-1, 1:])
        y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], yticklabels=[])
        x_hist = fig.add_subplot(grid[-1, 1:], xticklabels=[], yticklabels=[])

        main_ax.set_title("Box size distribution", fontsize=20)
        main_ax.set_xlabel("box width/image width", color='gray', fontsize=14,fontweight='bold')
        main_ax.set_ylabel("box height/image height", color='gray', fontsize=14,fontweight='bold')

        # divide axis by 10 parts
        main_ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        main_ax.yaxis.set_major_locator(plt.MaxNLocator(10))

        y_hist.set_xlabel('height frequency', fontweight='bold')
        y_hist.yaxis.set_major_locator(plt.MaxNLocator(10))
        x_hist.set_ylabel('width frequency',fontweight='bold')
        x_hist.xaxis.set_major_locator(plt.MaxNLocator(10))

        main_ax.plot(box_widths, box_heights, 'bo', markersize=3, alpha=0.3,color = 'deepskyblue')
        # main_ax.legend(['train', 'test'])

        x_hist.hist(box_widths, 700, histtype='stepfilled', orientation='vertical', color='deepskyblue')
        x_hist.invert_yaxis()
        y_hist.hist(box_heights, 700, histtype='stepfilled', orientation='horizontal', color='deepskyblue')
        y_hist.invert_xaxis()
        plt.show()

    def image_color_histogram(self, w: int = 12, h: int = 8, num_channels=3):
        """
        Plot image color channel's histogram
        :param w: plot width
        :param h: plot height
        :param num_channels: image depth (number of channels)
        :return:
        """
        channels = [np.zeros((256, 1)) for i in range(num_channels)]
        ch_labels = ['red', 'green', 'blue'] if num_channels == 3 else ['gray']
        num_images = 0
        for ann in tqdm(self._annotations):
            if not os.path.exists(ann.img_path):
                continue
            image = cv2.imread(ann.img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            num_channels = image.shape[2]
            if num_channels != 3:
                continue

            for j in range(num_channels):
                channels[j] += cv2.calcHist(image, [j], None, [256], [0, 256])

            num_images += 1

        if num_images != 0:
            for j in range(num_channels):
                channels[j] /= num_images
        else:
            raise ValueError('Image paths are not correct.')

        # create plot
        figure = plt.figure(figsize=(w, h))
        plt.title("Pixel channel distribution", fontsize=20)
        plt.axis("off")
        cols, rows = 2, 2
        for i in range(1, cols * rows + 1):
            figure.add_subplot(rows, cols, i)
            if i > num_channels + 1:
                break

            if i == num_channels + 1:
                for j in range(num_channels):
                    plt.plot(channels[j], color=ch_labels[j], alpha=0.8)
                plt.legend(ch_labels)
            else:
                plt.plot(channels[i - 1], color=ch_labels[i - 1], alpha=0.8)
                plt.legend([ch_labels[i - 1]])

            plt.ylabel("frequency", color='gray', fontsize=14, fontweight='bold')
            plt.xlim([0, 256])
            plt.ylim(ymin=0)

        plt.show()
