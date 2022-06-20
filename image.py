import cv2
import numpy as np
import os
import skimage.util
from skimage.exposure import rescale_intensity
from typing import Union


class Image:

    def __init__(self, data: np.ndarray = None, image_type: str = None, file_path: str = None):
        self._file_path: str = file_path
        self._data: np.ndarray = data
        self._image_type: str = image_type

    def __repr__(self):
        return "<%s (%s) %s %s>" % (self.__class__.__name__, self._image_type, self.shape, self.dtype)

    def __array__(self):
        return self._data

    def __setitem__(self, key, value):
        self._data.__setitem__(key, value)

    def __getitem__(self, item):
        self._data.__getitem__(item)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def image_type(self):
        return self._image_type

    @property
    def file_path(self):
        return os.path.normpath(self._file_path) if self._file_path is not None else ""

    @property
    def shape(self):
        return self._data.shape if self._data is not None else None

    @property
    def file_name(self):
        return os.path.splitext(os.path.basename(self.file_path))[0]

    @property
    def file_extension(self):
        return os.path.splitext(os.path.basename(self.file_path))[1]

    @property
    def data(self):
        return self._data

    def copy(self):
        return Image(data=self._data.copy() if self._data is not None else None,
                     image_type=self._image_type, file_path=self.file_path)

    def load(self, file_path: str):
        self._data = cv2.imread(file_path)
        self._image_type = "BGR"
        self._file_path = os.path.normpath(file_path)

    def save(self, file_path: str):
        cv2.imwrite(file_path, self._data)

    @staticmethod
    def _rotate_image_array(image: np.ndarray, angle: float):
        row, col = image.shape[:2]
        center = tuple(np.array([row, col]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_data = cv2.warpAffine(image, rot_mat, (col, row))
        return new_data

    def convert(self, image_type: str, copy: bool = True):
        assert self._data is not None, "No image data found to convert."
        assert self._image_type is not None, "Image type information not found."

        if self._image_type == image_type:
            return self.copy() if copy else self

        def _convert(data, in_type, out_type):
            conversion_string = "COLOR_" + in_type + "2" + out_type
            conversion_code = getattr(cv2, conversion_string)
            return cv2.cvtColor(data, conversion_code)

        out_data = _convert(self._data, self._image_type, image_type)

        if copy:
            return Image(data=out_data, image_type=image_type)
        else:
            self._data = out_data
            return self

    def astype(self, dtype: str, copy: bool = True):
        conversion_function = getattr(skimage.util, "img_as_" + dtype)
        data = conversion_function(self._data, force_copy=copy)
        if copy:
            return Image(data=data, image_type=self._image_type)
        else:
            self._data = data
            return self

    def rescale_intensity(self, in_range: Union[str, tuple] = 'image', out_range: Union[str, tuple] = 'dtype',
                          copy: bool = True):
        # noinspection PyTypeChecker
        data = rescale_intensity(image=self._data, in_range=in_range, out_range=out_range)
        if copy:
            # noinspection PyTypeChecker
            return Image(data=data, image_type=self._image_type)
        else:
            self._data = data
            return self

    @staticmethod
    def adjust_brightness(img: np.ndarray, value):
        num_channels = 1 if len(img.shape) < 3 else 1 if img.shape[-1] == 1 else 3
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if num_channels == 1 else img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            value = int(-value)
            lim = 0 + value
            v[v < lim] = 0
            v[v >= lim] -= value
        final_hsv = cv2.merge((h, s, np.array(v, dtype="uint8")))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if num_channels == 1 else img
        return img

    def resize(self, resolution: int = 500000):
        if self._data is None:
            return
        # data_resolution = self.shape[0] * self.shape[1]
        # area_scale = resolution / data_resolution
        h_by_w = self.shape[0] / self.shape[1]
        wd = int(np.sqrt(resolution / h_by_w))
        hd = int(resolution / wd)
        new_data = cv2.resize(self._data, (wd, hd))
        return Image(data=new_data, image_type=self._image_type)
