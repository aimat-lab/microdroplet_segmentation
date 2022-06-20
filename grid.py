import numpy as np
import yaml
import os
import cv2
import scipy as sp
import scipy.signal


class Grid:

    min_expected_segments = 3
    max_possible_segments = 1000
    default_origin = np.array([0, 0], dtype="int")  # is [y, x]
    default_delta_x = 50
    default_delta_y = 50
    default_num_x = np.array([0, 4], dtype="int")  # is [left, right] >= 0
    default_num_y = np.array([0, 4], dtype="int")  # is [left, right] >= 0

    def __init__(self):
        self.file_path = None
        self.origin = None  # is [y, x]
        self.delta_x = None
        self.delta_y = None
        self.num_x = None  # is [left, right] >= 0
        self.num_y = None  # is [left, right] >= 0

    @property
    def is_valid(self):
        return all([x is not None for x in ["origin", "delta_x", "delta_y", "num_x", "num_y"]])

    @staticmethod
    def save_yaml_file(out, file_name):
        with open(file_name, 'w') as yaml_file:
            yaml.dump(out, yaml_file, default_flow_style=False)

    @staticmethod
    def load_yaml_file(file_name):
        with open(file_name, 'r') as stream:
            out = yaml.safe_load(stream)
        return out

    def make_grid(self, rescale_x: float = 1.0, rescale_y: float = 1.0):
        grid_x_pos = np.arange(-self.num_x[0], self.num_x[1]) * self.delta_x + self.origin[1]
        grid_y_pos = np.arange(-self.num_y[0], self.num_y[1]) * self.delta_y + self.origin[0]
        return grid_x_pos*rescale_x, grid_y_pos*rescale_y

    def optimize_grid(self, image_intensity: np.ndarray):

        def opt_axis(gp: np.ndarray, delta, sl: np.ndarray, axis: int):
            ra = np.arange(delta)
            shift = np.expand_dims(ra, axis=-1) + np.expand_dims(gp, axis=0) - np.amin(gp)
            shift[shift >= image_intensity.shape[axis]] = 0
            shift = np.array(shift, dtype="int")
            # print(np.argmin(np.sum(self.slice_x[sx],axis=-1)))
            offset = ra[np.argmin(np.sum(sl[shift], axis=-1))]
            gp = gp + offset
            return gp

        slice_x = np.sum(image_intensity, axis=0)
        slice_y = np.sum(image_intensity, axis=1)
        grid_x_pos, grid_y_pos = self.make_grid()
        grid_x_pos = opt_axis(grid_x_pos, self.delta_x, slice_x, 1)
        grid_y_pos = opt_axis(grid_y_pos, self.delta_y, slice_y, 0)
        self.origin = np.array([grid_y_pos[self.num_y[0]], grid_x_pos[self.num_x[0]]])
        return grid_x_pos, grid_y_pos

    def confine_grid_to_figure(self):
        pass

    @staticmethod
    def _get_fft_main_frequency_1d(signal, min_f, max_f):
        """Return the main frequency of a 1-D signal."""
        sum_y = signal
        f_sum_y = np.fft.rfft(sum_y - np.min(sum_y))
        freq_f_sum_y = np.fft.rfftfreq(sum_y.shape[-1])
        selection = np.logical_and(freq_f_sum_y > min_f, freq_f_sum_y < max_f)
        sel_freq_fy = freq_f_sum_y[selection]
        sel_fy = f_sum_y[selection]
        return sel_freq_fy[np.argmax(np.abs(sel_fy))]

    def find_grid_spacing_fft(self, image: np.ndarray,
                              min_expected_segments=1, max_possible_segments=1000):
        """Estimate spacing by main frequency component in x,y-slices."""
        # Main frequency for each direction
        sl_x = np.sum(image, axis=0)
        sl_y = np.sum(image, axis=1)
        fy = self._get_fft_main_frequency_1d(
            sl_y, 1.0 / image.shape[0] * min_expected_segments, 1 / image.shape[0] * max_possible_segments)
        fx = self._get_fft_main_frequency_1d(
            sl_x, 1.0 / image.shape[1] * min_expected_segments, 1 / image.shape[1] * max_possible_segments)
        return 1 / fx, 1 / fy

    @staticmethod
    def find_peak_position_slices(image: np.ndarray,
                                  find_kwargs_x: dict = None,
                                  find_kwargs_y: dict = None):
        find_kwargs_x = {} if find_kwargs_x is None else find_kwargs_x
        find_kwargs_y = {} if find_kwargs_y is None else find_kwargs_y
        sl_x = np.sum(image, axis=0)
        sl_y = np.sum(image, axis=1)
        x_peaks, _ = sp.signal.find_peaks(sl_x, **find_kwargs_x)
        y_peaks, _ = sp.signal.find_peaks(sl_y, **find_kwargs_y)
        return x_peaks, y_peaks

    def propose_grid(self, image_intensity: np.ndarray):
        """Main function to run the grid for segmentation."""
        dx, dy = self.find_grid_spacing_fft(image_intensity, min_expected_segments=self.min_expected_segments,
                                            max_possible_segments=self.max_possible_segments)
        self.origin = np.array([0, 0])
        self.delta_x = dx
        self.delta_y = dy
        self.num_x = np.array([0, int(image_intensity.shape[1] / dx)], dtype="int")
        self.num_y = np.array([0, int(image_intensity.shape[0] / dy)], dtype="int")
        self.optimize_grid(image_intensity)
        return self.make_grid()

    def save(self, directory_path, file_name="GridProperties.yaml"):
        self.file_path = os.path.normpath(os.path.join(directory_path, file_name))
        self.save_yaml_file({"grid_reference": str(self.file_path),
                             "origin": [int(x) for x in self.origin],
                             "delta_x": float(self.delta_x),
                             "delta_y": float(self.delta_y),
                             "num_x": [int(x) for x in self.num_x],
                             "num_y": [int(x) for x in self.num_y]},
                            self.file_path)

    def load(self, directory_path: str, file_name: str = "GridProperties.yaml"):
        self.file_path = os.path.normpath(os.path.join(directory_path, file_name))
        grid_dict = self.load_yaml_file(self.file_path)
        for x in ["origin", "delta_x", "delta_y", "num_x", "num_y"]:
            setattr(self, x, grid_dict[x])

    def reset_default_grid(self):
        self.origin = self.default_origin  # is y,x
        self.delta_x = self.default_delta_x
        self.delta_y = self.default_delta_y
        self.num_x = self.default_num_x  # is left, right
        self.num_y = self.default_num_y  # is left, right

    def shift_origin(self, nx=None, ny=None):
        if nx is not None:
            index_shift = np.arange(-self.num_x[0], self.num_x[1])[nx]
            self.origin[1] += index_shift * self.delta_x
            self.num_x -= np.array([-index_shift, index_shift], dtype="int")
        if ny is not None:
            index_shift = np.arange(-self.num_y[0], self.num_y[1])[ny]
            self.origin[0] += index_shift * self.delta_y
            self.num_y -= np.array([-index_shift, index_shift], dtype="int")

    def add_grid_row(self, ny: np.ndarray):
        self.num_y += ny
        self.num_y[self.num_y < 0] = 0

    def add_grid_column(self, nx: np.ndarray):
        self.num_x += nx
        self.num_x[self.num_x < 0] = 0

    def apply_offset(self, x=None, y=None):
        if x is not None:
            self.origin[1] += x
        if y is not None:
            self.origin[0] += y

    def apply_stretch(self, dx=None, dy=None):
        if dx is not None:
            self.delta_x += dx
        if dy is not None:
            self.delta_y += dy


class ImageGrid:

    def __init__(self, image: np.ndarray, grid_x_pos: np.ndarray, grid_y_pos: np.ndarray):
        self.image, self._grid_shape, self._grid_dxy = self.make_grid_image(
            image, grid_x_pos, grid_y_pos)

    @staticmethod
    def make_grid_image(image: np.ndarray, grid_x_pos: np.ndarray, grid_y_pos: np.ndarray):
        """Make new image with equal sized segments of grid."""
        assert image is not None, "Missing image to make image grid."
        assert len(grid_x_pos) > 0 and len(grid_y_pos) > 0
        delta_x = np.ceil(np.mean(grid_x_pos[1:] - grid_x_pos[:-1]))
        delta_y = np.ceil(np.mean(grid_y_pos[1:] - grid_y_pos[:-1]))
        dxy = [int(delta_x), int(delta_y)]
        pos_x = np.maximum(np.floor(grid_x_pos).astype("int"), np.zeros_like(grid_x_pos, dtype="int"))
        pos_y = np.maximum(np.floor(grid_y_pos).astype("int"), np.zeros_like(grid_y_pos, dtype="int"))
        channels = [] if len(image.shape) < 3 else [image.shape[-1]]
        image_grid = np.zeros([len(pos_y) - 1, len(pos_x) - 1, dxy[1], dxy[0]] + channels, image.dtype)
        for i in range(len(pos_y) - 1):
            for j in range(len(pos_x) - 1):
                image_box = image[pos_y[i]:pos_y[i + 1], pos_x[j]:pos_x[j + 1]]
                image_grid[i, j, :image_box.shape[0], :image_box.shape[1], ...] = image_box
        grid = np.concatenate([np.concatenate(x, axis=1) for x in image_grid], axis=0)
        grid_shape = [len(pos_x) - 1, len(pos_y) - 1]
        return grid, grid_shape, dxy

    def __getitem__(self, item):
        i, j = item
        dx, dy = self._grid_dxy
        return self.image[dy * j:dy * (j + 1), dx * i:dx * (i + 1)]

    def __setitem__(self, key, value):
        i, j = key
        dx, dy = self._grid_dxy
        self.image[dy * j:dy * (j + 1), dx * i:dx * (i + 1)] = value

    @property
    def shape(self):
        return np.array(self.image.shape)

    @property
    def grid_shape(self):
        return np.array(self._grid_shape)

    @property
    def dx(self):
        return int(self._grid_dxy[0])

    @property
    def dy(self):
        return int(self._grid_dxy[1])

    def save(self, directory_path, file_name: str = "ImageGrid.jpg", font: int = 1, f_size: int = 1):
        image = self.image.copy()
        if len(image.shape) == 2:  # Gray scale image
            image = (image.astype("float") - np.amin(image)) / np.amax(image) * 255
            image = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_GRAY2BGR)
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                cv2.putText(
                    image, "{0}/{1}".format(i, j), (self.dx * i, self.dy * j + 15), font, f_size, (255, 255, 255), 2,
                    bottomLeftOrigin=False)
        for i in range(self.grid_shape[0]):
            image[:, self.dx * i, :] = np.array([[255, 255, 255]], dtype="uint8")
        for i in range(self.grid_shape[1]):
            image[self.dy * i, :, :] = np.array([[255, 255, 255]], dtype="uint8")
        cv2.imwrite(os.path.join(directory_path, file_name), image)

    @property
    def grid_x_pos(self):
        return np.arange(0, self._grid_shape[0]) * self._grid_dxy[0]

    @property
    def grid_y_pos(self):
        return np.arange(0, self._grid_shape[1]) * self._grid_dxy[1]

    def resize(self, factor: float = 0.35):
        if self.image is None:
            return
        self._grid_dxy = np.array(np.array(self._grid_dxy, dtype="float") * factor, dtype="int")
        wd = self._grid_dxy[0] * self._grid_shape[0]
        hd = self._grid_dxy[1] * self._grid_shape[1]
        self.image = cv2.resize(self.image, (wd, hd))
        return self
