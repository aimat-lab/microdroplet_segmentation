import cv2
import numpy as np
import os


class Image:

    _image_type_list = ["BRG", "RGB", "GRAY", "GRAYSCALE"]

    def __init__(self, data: np.ndarray = None, image_type: str = None, file_path: str = None):
        self._file_path = file_path
        self._image = data
        self._image_type = image_type

    def copy(self):
        return Image(data=self._image.copy() if self._image is not None else None,
                     image_type=self._image_type, file_path=self.file_path)

    @property
    def file_path(self):
        return os.path.normpath(self._file_path) if self._file_path is not None else ""

    @property
    def shape(self):
        return self._image.shape

    @property
    def file_name(self):
        return os.path.splitext(os.path.basename(self.file_path))[0]

    @property
    def file_extension(self):
        return os.path.splitext(os.path.basename(self.file_path))[1]

    def load_image(self, file_path: str):
        self._image = cv2.imread(file_path)
        self._image_type = "BGR"
        self._file_path = os.path.normpath(file_path)

    def data(self):
        return self._image

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float):
        row, col = image.shape[:2]
        center = tuple(np.array([row, col]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_image = cv2.warpAffine(image, rot_mat, (col, row))
        return new_image

    def convert(self, image_type: str = "RGB"):
        print(self._image_type, image_type)
        assert self._image is not None, "No image data found to convert."
        assert self._image_type is not None, "Image type information not found."
        if self._image_type == image_type:
            return self.copy()

        if self._image_type == "BGR" and image_type == "GRAYSCALE":
            assert len(self._image.shape) == 3, "Image image_type '%s' does not match channels." % self._image_type
            gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
            return Image(data=gray/np.amax(gray), image_type="GRAYSCALE")  # should be float

    @staticmethod
    def adjust_brightness(image: np.ndarray, value):
        num_channels = 1 if len(image.shape) < 3 else 1 if image.shape[-1] == 1 else 3
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if num_channels == 1 else image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if num_channels == 1 else image
        return image