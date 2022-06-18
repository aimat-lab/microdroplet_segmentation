import cv2
import numpy as np
import os
# import scipy as sp
import pandas as pd
# import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
# import sys
import argparse
from skimage import morphology
# from skimage.feature import canny
from skimage import segmentation
from scipy import ndimage as ndi
from skimage.filters import sobel, gaussian
# from skimage.filters import median
# from skimage.measure import find_contours
# from skimage import data, img_as_float
# from skimage import exposure
# import time
# import yaml
from grid import Grid
from image import Image

mpl.use("Qt5Cairo")
mpl.rcParams["keymap.back"] = ['backspace']
mpl.rcParams["keymap.forward"] = []
mpl.rcParams["keymap.save"] = ['ctrl+s']  # Remove s here
mpl.rcParams["keymap.home"] = ['h', 'home']


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
        pos_x = np.maximum(np.array(np.floor(grid_x_pos), dtype="int"), np.zeros_like(grid_x_pos, dtype="int"))
        pos_y = np.maximum(np.array(np.floor(grid_y_pos), dtype="int"), np.zeros_like(grid_y_pos, dtype="int"))
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

    def plot(self):
        plt.figure()
        plt.imshow(self.image)
        plt.show()

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

    def save(self, directory_path, file_name: str = "ImageGrid.png", font: int = 1, f_size: int = 1):
        image = self.image.copy()
        if len(image.shape) == 2:  # Gray scale image
            image = np.array(image / np.amax(image) * 255, dtype="uint8")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        print(image.shape)
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                cv2.putText(
                    image, "{0}/{1}".format(i, j), (self.dx * i, self.dy * j + 15), font, f_size, (255, 255, 255), 2,
                    bottomLeftOrigin=False)
        for i in range(self.grid_shape[0]):
            image[:, self.dx * i, :] = np.array([[255, 255, 255]], dtype="uint8")
        for i in range(self.grid_shape[1]):
            image[self.dy * i, :, :] = np.array([[255, 255, 255]], dtype="uint8")
        print(cv2.imwrite(os.path.join(directory_path, file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))

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


class DropletSeparation:

    def __init__(self, image: Image, grid: Grid):
        self.original_image = image
        self.original_grid = grid

        self.grid = ImageGrid(image.convert("GRAY").astype("float").rescale_intensity().data,
                              *grid.make_grid())
        self.grid_show = ImageGrid(image.data, *grid.make_grid())
        self.grid_edges = ImageGrid(image.convert("GRAY").astype("float").rescale_intensity().data,
                                    *grid.make_grid())
        self.grid_segments = ImageGrid(image.convert("GRAY").astype("float").rescale_intensity().data,
                                       *grid.make_grid())
        self.grid_edges_dilated = ImageGrid(image.convert("GRAY").astype("float").rescale_intensity().data,
                                            *grid.make_grid())

        # Preview image
        self.grid_preview = ImageGrid(image.data, *grid.make_grid()).resize()
        self.grid_edges_dilated_preview = ImageGrid(image.convert("GRAY").astype("float").rescale_intensity().data,
                                                    *grid.make_grid()).resize()

        self.grid_edges_dilated_preview.image = np.array(self.grid_edges_dilated_preview.image, dtype="bool")
        self.grid_edges_dilated.image = np.array(self.grid_edges_dilated.image, dtype="bool")

        self.mode_param_default = np.array([[[0, 0.3, 0.5, 5, 64, 3, 0.1]]])
        self.mode_params_step = np.array([[[0, 0.1, 0.1, 0.25, 5, 1, 0.1]]])
        self.mode_param_label = {1: "Marker background level",
                                 2: "Marker droplet level",
                                 3: "Sigma",
                                 4: "Min. Drop Size",
                                 5: "Median Kernel",
                                 6: "Min. Intensity level"}
        self.mode_params = np.zeros(list(self.grid.grid_shape) + [7])
        self.mode_params[:, :] = self.mode_param_default

    @staticmethod
    def find_max_class(labeled_array: np.ndarray):
        labels, counts = np.unique(labeled_array, return_counts=True)
        labels_0 = labels[labels != 0]
        counts_0 = counts[labels != 0]
        if len(labels_0) == 0:
            return -1
        return labels_0[np.argmax(counts_0)]

    @staticmethod
    def find_binary_step(arr):
        ls = np.logical_or(arr[:-1, :] == arr[1:, :] - 1, arr[:-1, :] == arr[1:, :] + 1)
        rs = np.logical_or(arr[:, :-1] == arr[:, 1:] - 1, arr[:, :-1] == arr[:, 1:] + 1)
        out = np.zeros_like(arr, dtype="uint8")
        out[:-1, :] = np.logical_or(out[:-1, :], ls)
        out[1:, :] = np.logical_or(out[1:, :], ls)
        out[:, :-1] = np.logical_or(out[:, :-1], rs)
        out[:, 1:] = np.logical_or(out[:, 1:], rs)
        return out

    def segmentation_watershed(self, pic: np.ndarray, min_cut: float = 0.3, max_cut: float = 0.5, sigma: float = 5,
                               min_drop: int = 64, median_kernel_size: int = 3, min_intensity: float = 0.1):
        # print(min_cut, max_cut, sigma, min_drop, median_kernel_size, min_intensity)
        # restrict min/max marker
        min_cut = max(min_cut, 0.01)  # must be larger than 0
        min_cut = min(min_cut, 0.99)
        max_cut = min(max_cut, 1)

        pic_smooth = cv2.medianBlur(np.array(pic * 255, dtype="uint8"), ksize=max(int(median_kernel_size), 1)) / 255
        pic_smooth = gaussian(pic_smooth, sigma=max(sigma, 0))
        pic_preproc = (pic_smooth - np.amin(pic_smooth)) / np.amax(pic_smooth)

        elevation_map = sobel(pic_preproc)
        markers = np.zeros_like(pic)
        markers[pic_preproc < min_cut] = 1
        markers[pic_preproc >= max_cut] = 2
        segmentation_drops = segmentation.watershed(elevation_map, markers) == 2
        drops_cleaned = morphology.remove_small_objects(segmentation_drops, min_drop)
        labeled_drops, _ = ndi.label(drops_cleaned)
        max_cl = self.find_max_class(labeled_drops)
        max_drop = labeled_drops == max_cl
        if np.max(pic) < min_intensity:
            max_drop = np.zeros_like(max_drop)

        edges = self.find_binary_step(max_drop)
        max_edge = np.logical_and(edges, max_drop)

        return max_drop, max_edge

    def find_segmentation(self):
        for i in range(self.grid.grid_shape[0]):
            for j in range(self.grid.grid_shape[1]):
                self.find_segmentation_index(i, j)

    def find_segmentation_index(self, i, j):
        max_drop, max_edge = self.segmentation_watershed(self.grid[i, j], *self.mode_params[i, j, 1:])
        self.grid_edges[i, j] = max_edge
        self.grid_segments[i, j] = max_drop
        max_edge_dilated = np.array(
            cv2.dilate(np.array(max_edge, dtype="float32"), np.ones((3, 3))), dtype="bool")
        self.grid_edges_dilated[i, j] = max_edge_dilated
        reduced_shape = self.grid_edges_dilated_preview[i, j].shape
        reduced = cv2.resize(np.array(max_edge_dilated, dtype="float32"), (reduced_shape[1], reduced_shape[0]))
        self.grid_edges_dilated_preview[i, j] = np.array(reduced, dtype="bool")


class GUI:
    brightness_increase = 10

    def __init__(self, droplet: DropletSeparation):
        self.droplet = droplet
        self.image = droplet.grid_show
        self.image_preview = droplet.grid_preview

        self.fig = None
        self.ax = None
        self.image_in_fig = None
        self.fig_x_lines = []
        self.fig_y_lines = []
        self.y_label_text = ""
        self.log_info = ["> Event log:\n"]
        self.log_text = None
        self.bright = 0

        self.mode_preview = False
        self.mode_param_selection = 1
        self.mode_param_label = droplet.mode_param_label
        self.debug = False

    def add_logg(self, info):
        max_len_log = 40
        if len(self.log_info) > max_len_log:
            self.log_info = self.log_info[-max_len_log:]
        self.log_info.append(info + "\n")

    def _draw_segmentation(self, image, edges, preview=False, flush=True):
        if self.mode_preview != preview:
            # Reset ax lim also
            self.ax.set_xlim((0, image.shape[1]))
            self.ax.set_ylim((image.shape[0], 0))
            self.mode_preview = preview
        title_preview = "PREVIEW, " if preview else ""
        self.ax.set_title(
             title_preview + "Parameter: " + self.mode_param_label[self.mode_param_selection])
        image_array = image.image.copy()
        image_array = Image.adjust_brightness(image_array, self.bright)
        # cmap = plt.get_cmap('hot')
        # img = cmap(img)
        if self.image_in_fig is not None:
            self.image_in_fig.remove()
        image_array[edges.image] = np.array([[0, 255, 0]])
        self.image_in_fig = self.ax.imshow(image_array, vmax=self.bright)
        for i, lxy in enumerate(self.fig_y_lines):
            lxy.set_ydata((image.grid_y_pos[i], image.grid_y_pos[i]))
        for i, lxy in enumerate(self.fig_x_lines):
            lxy.set_xdata((image.grid_x_pos[i], image.grid_x_pos[i]))
        self.log_text.set_text("".join(self.log_info))
        # self.log_text.set_position((self.rgb.shape[1]*1.02, self.rgb.shape[0]))
        self.log_text.set_position((1.02, 0))
        if flush:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def draw_segmentation_preview(self, flush=True):
        self._draw_segmentation(self.image_preview, self.droplet.grid_edges_dilated_preview, preview=True, flush=flush)

    def draw_segmentation(self, flush=True):
        self._draw_segmentation(self.image, self.droplet.grid_edges_dilated, preview=False, flush=flush)

    @staticmethod
    def _find_grid_segment(image, event):
        diffx = -image.grid_x_pos + event.xdata
        diffy = -image.grid_y_pos + event.ydata
        x_idx = np.argmin(np.where(diffx > 0, diffx, np.inf))
        y_idx = np.argmin(np.where(diffy > 0, diffy, np.inf))
        return x_idx, y_idx

    def key_press_event(self, event):
        # print('you pressed', event.key, "at", event.xdata, event.ydata)
        if event.key == "enter":
            print("Accept current grid segmentation...")
            self.fig.canvas.stop_event_loop()

        elif event.key == "r":
            self.draw_segmentation()

        elif event.key == "down":
            ms = self.mode_param_selection
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                if self.mode_preview:
                    i, j = self._find_grid_segment(self.image_preview, event)
                else:
                    i, j = self._find_grid_segment(self.image, event)
                self.droplet.mode_params[i, j, ms] -= self.droplet.mode_params_step[0, 0, ms]
                self.add_logg(
                    "> " + self.mode_param_label[ms] + " for ({0}, {1}) to {2}".format(
                        i, j, self.droplet.mode_params[i, j, ms]))
                self.droplet.find_segmentation_index(i, j)
                self.draw_segmentation_preview()
            else:
                self.droplet.mode_params[:, :, ms] -= self.droplet.mode_params_step[:, :, ms]
                self.add_logg("> All " + self.mode_param_label[ms] + " by -{0}".format(
                    self.droplet.mode_params_step[0, 0, ms]))
                self.draw_segmentation_preview()
        elif event.key == "up":
            ms = self.mode_param_selection
            if event.inaxes and event.xdata is not None and event.ydata is not None:
                if self.mode_preview:
                    i, j = self._find_grid_segment(self.image_preview, event)
                else:
                    i, j = self._find_grid_segment(self.image, event)
                self.droplet.mode_params[i, j, ms] += self.droplet.mode_params_step[0, 0, ms]
                self.add_logg(
                    "> " + self.mode_param_label[ms] + " for ({0}, {1}) to {2}".format(
                        i, j, self.droplet.mode_params[i, j, ms]))
                self.droplet.find_segmentation_index(i, j)
                self.draw_segmentation_preview()
            else:
                self.droplet.mode_params[:, :, ms] += self.droplet.mode_params_step[:, :, ms]
                self.droplet.find_segmentation()
                self.add_logg("> All" + self.mode_param_label[ms] + "by +{0}".format(
                    self.droplet.mode_params_step[0, 0, ms]))
                self.draw_segmentation_preview()
        elif event.key == "1":
            self.mode_param_selection = 1
            self.add_logg("> Switch parameter to {0}".format(self.mode_param_label[self.mode_param_selection]))
            self.draw_segmentation_preview()
        elif event.key == "2":
            self.mode_param_selection = 2
            self.add_logg("> Switch parameter to {0}".format(self.mode_param_label[self.mode_param_selection]))
            self.draw_segmentation_preview()
        elif event.key == "3":
            self.mode_param_selection = 3
            self.add_logg("> Switch parameter to {0}".format(self.mode_param_label[self.mode_param_selection]))
            self.draw_segmentation_preview()
        elif event.key == "4":
            self.mode_param_selection = 4
            self.add_logg("> Switch parameter to {0}".format(self.mode_param_label[self.mode_param_selection]))
            self.draw_segmentation_preview()
        elif event.key == "5":
            self.mode_param_selection = 5
            self.add_logg("> Switch parameter to {0}".format(self.mode_param_label[self.mode_param_selection]))
            self.draw_segmentation_preview()
        elif event.key == "6":
            self.mode_param_selection = 6
            self.add_logg("> Switch parameter to {0}".format(self.mode_param_label[self.mode_param_selection]))
            self.draw_segmentation_preview()
        elif event.key == "-":
            self.bright -= self.brightness_increase
            self.add_logg("> Change brightness to {}".format(self.bright))
            self.draw_segmentation_preview()
        elif event.key == "+":
            self.bright += self.brightness_increase
            self.add_logg("> Change brightness to {}".format(self.bright))
            self.draw_segmentation_preview()
        elif event.key == "m":
            self.add_logg("> Average Parameter Stats:")
            for key, item in self.droplet.mode_param_label.items():
                self.add_logg("> {0}: <{1}>".format(item, np.mean(self.droplet.mode_params[:, :, int(key)])))
            self.draw_segmentation_preview()

    def button_press_event(self, event):
        pass

    def _add_grid_numbers(self):
        for i, x in enumerate(self.image.grid_x_pos):
            for j, y in enumerate(self.image.grid_y_pos):
                plt.text(x, y, "{0}/{1}".format(i, j), color="w",
                         horizontalalignment='left',
                         verticalalignment='top')

    def export_data(self, filepath):
        cols_line = self.grid_x_sep
        rows_line = self.grid_y_sep
        pixel_size = []
        pixel_int = []
        for i in range(len(rows_line) - 1):
            row = []
            row_int = []
            for j in range(len(cols_line) - 1):
                segment_ij = self.grid_segments[rows_line[i]:rows_line[i + 1], cols_line[j]:cols_line[j + 1]]
                segment_int = self.gray[rows_line[i]:rows_line[i + 1], cols_line[j]:cols_line[j + 1]]
                row.append(np.sum(segment_ij))
                row_int.append(np.sum(segment_int[np.array(segment_ij, dtype="bool")]))
            pixel_size.append(row)
            pixel_int.append(row_int)
        df = pd.DataFrame(np.array(pixel_size))
        df.to_excel(os.path.join(filepath, "DropletsSize.xls"))
        df = pd.DataFrame(np.array(pixel_int))
        df.to_excel(os.path.join(filepath, "DropletsIntensity.xls"))
        if os.path.exists(os.path.join(filepath, "scale_bar_length.txt")):
            with open(os.path.join(filepath, "scale_bar_length.txt"), "r") as f:
                scale = int(f.read().strip())
            if isinstance(scale, int):
                if scale > 0:
                    df2 = pd.DataFrame(np.array(pixel_size) / scale / scale)
                    df2.to_excel(os.path.join(filepath, "DropletsSizeScaled.xls"))
            else:
                print("Could not read int")

    def run(self, window_title: str = "Droplet Segmentation"):
        plt.ion()
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.y_label_text = "".join(["Press: 1,2, etc. to select paramter.\n"
                                     "Change (move Mouse): Press 'up', 'down' for change parameter.\n",
                                     "Press: 'r' to render at max resolution.\n",
                                     "Press: '+', '-' to change brightness.\n",
                                     "Press: 'm' to get current mean settings."
                                     ])
        plt.xlim([0, self.image.shape[1]])
        plt.ylim([self.image.shape[0], 0])
        plt.ylabel(self.y_label_text, rotation='horizontal', ha='right')
        self.log_text = plt.text(1.02, 0, "".join(self.log_info), backgroundcolor='w', transform=ax.transAxes)
        for i in self.image_preview.grid_y_pos:
            lx = self.ax.axhline(y=i, color='r', linestyle='-', lw=0.5)
            self.fig_y_lines.append(lx)
        for j in self.image_preview.grid_x_pos:
            ly = self.ax.axvline(x=j, color='r', linestyle='-', lw=0.5)
            self.fig_x_lines.append(ly)
        self.draw_segmentation_preview(flush=False)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()
        fig_manager.set_window_title(window_title)
        fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        fig.canvas.mpl_connect('button_press_event', self.button_press_event)
        fig.canvas.start_event_loop()
        fig.clear()
        plt.close(fig)
        plt.close("all")


if __name__ == "__main__":
    # Input arguments from command line.
    parser = argparse.ArgumentParser(description='Run DropletSeparation.')
    parser.add_argument("--file", required=False, help="Input filepath of image.")
    args = vars(parser.parse_args())
    print("Input of argparse:", args)

    # File and path information
    # arg_file_path = args["file"]
    arg_file_path = "output/HG2A_30s/HG2A_30s_select.jpg"
    arg_result_path = os.path.dirname(arg_file_path)
    arg_file_name = os.path.basename(arg_file_path)

    # Load Image
    img = Image()
    img.load_image(arg_file_path)

    # Make Grid
    grd = Grid()
    grd.load(arg_result_path)

    # Image Grid
    seg = DropletSeparation(img, grd)
    seg.find_segmentation()

    # Propose Grid
    gi = GUI(seg)
    gi.run()

    # Export results
    seg.grid_segments.save(arg_result_path)
