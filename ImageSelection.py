import cv2
import numpy as np
import os
import argparse
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

mpl.use("Qt5Cairo")
mpl.rcParams["keymap.back"] = ['backspace']
mpl.rcParams["keymap.forward"] = []


class Image:

    def __init__(self, image: np.ndarray = None):
        self.rgb = image
        self.gray_norm = None
        self.file_path = None
        self.rotation = 0
        self.range_x = None
        self.range_y = None

    def compute_gray_image(self):
        if self.rgb is not None:
            gray = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)
            self.gray_norm = gray / np.amax(gray)

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float):
        row, col = image.shape[:2]
        center = tuple(np.array([row, col]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_image = cv2.warpAffine(image, rot_mat, (col, row))
        return new_image

    @property
    def shape(self):
        return self.rgb.shape

    @property
    def file_name(self):
        return os.path.splitext(os.path.basename(self.file_path))[0]

    @property
    def file_extension(self):
        return os.path.splitext(os.path.basename(self.file_path))[1]

    @staticmethod
    def save_yaml_file(out, file_name):
        with open(file_name, 'w') as yaml_file:
            yaml.dump(out, yaml_file, default_flow_style=False)

    def load_image(self, file_path: str):
        self.rgb = cv2.imread(file_path)
        self.file_path = os.path.realpath(file_path)

    def rotate_step(self, step):
        self.rotation = self.rotation + step
        self.gray_norm = self.rotate_image(self.gray_norm, step)

    def set_view(self, span_x, span_y):
        rng_x = np.sort(np.array(span_x, dtype="int"))
        rng_y = np.sort(np.array(span_y, dtype="int"))
        print(rng_x, rng_y)
        rng_x[rng_x < 0] = 0
        rng_y[rng_y < 0] = 0
        self.range_x = rng_x
        self.range_y = rng_y

    def export(self, dir_path: str, file_name: str = None):
        if file_name is None:
            file_name = self.file_name+"_select"+self.file_extension
        new_rng = self.rgb.copy()
        new_rng = self.rotate_image(new_rng, self.rotation)
        new_rng = new_rng[self.range_y[0]:self.range_y[1], self.range_x[0]:self.range_x[1]]
        print(cv2.imwrite(os.path.join(dir_path, file_name), new_rng))
        self.save_yaml_file({"file_path": self.file_path,
                             "rotation": float(self.rotation),
                             "range_x": [int(x) for x in self.range_x],
                             "range_y": [int(x) for x in self.range_y]
                             }, os.path.join(dir_path, "ImageSelection.yaml"))


class GUI:

    def __init__(self, image: Image = None):

        self.image = image
        self.image.compute_gray_image()

        self.brightness_increase = 0.1
        self.img_rotation_step = 0.25
        self.bright = 1.0

        self.fig = None
        self.ax = None
        self.image_in_fig = None

    def _redraw_image(self):
        scale_x, scale_y = self.ax.get_xlim(), self.ax.get_ylim()
        self.image_in_fig.remove()
        self.image_in_fig = self.ax.imshow(self.image.gray_norm, cmap='hot', vmax=self.bright)
        self.ax.set_xlim(scale_x)
        self.ax.set_ylim(scale_y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def key_press_event(self, event):
        print('you pressed', event.key, "at", event.xdata, event.ydata)
        if event.key == "enter":
            print("Accept current grid segmentation...")
            self.fig.canvas.stop_event_loop()

        elif event.key == "left":
            self.image.rotate_step(self.img_rotation_step)
            self._redraw_image()

        elif event.key == "right":
            self.image.rotate_step(-self.img_rotation_step)
            self._redraw_image()

        elif event.key == "-":
            self.bright += self.brightness_increase
            self.bright = min(self.bright, 1)
            self._redraw_image()

        elif event.key == "+":
            self.bright -= self.brightness_increase
            self.bright = max(self.bright, self.brightness_increase)
            self._redraw_image()

    def button_press_event(self, event):
        pass

    def propose(self):
        # fig = plt.figure()
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        self.image_in_fig = ax.imshow(self.image.gray_norm, cmap='hot', vmax=self.bright)
        Cursor(ax, horizOn=True, vertOn=True, color='red', linewidth=1, useblit=True)
        plt.ion()
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        info_cap = "".join(["Press 'left' or 'right' to rotate.\n",
                            "Pick magnifying glass from menu to select ROI.\n",
                            "Press 'g' for additional static gridlines.\n",
                            "Press '+', '-' to increase brightness (view only).\n"
                            ])
        plt.ylabel(info_cap, rotation='horizontal', ha='right')
        plt.show()
        fig_manager.set_window_title(self.image.file_name)
        fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        fig.canvas.mpl_connect('button_press_event', self.button_press_event)
        fig.canvas.start_event_loop()
        self.image.set_view(ax.get_xlim(), ax.get_ylim())
        fig.clear()
        plt.close(fig)
        plt.close("all")


if __name__ == "__main__":
    # Input arguments from command line.
    parser = argparse.ArgumentParser(description='Image selection.')
    parser.add_argument("--file", required=True, help="Input filepath of image.")
    args = vars(parser.parse_args())
    print("Input of argparse:", args)

    # File and path information
    arg_file_path = args["file"]
    # arg_file_path = "output/HG2A_30s/HG2A_30s.jpg"
    arg_result_path = os.path.dirname(arg_file_path)
    arg_file_name = os.path.basename(arg_file_path)

    # Load Image
    img = Image()
    img.load_image(arg_file_path)

    # Propose Grid
    gi = GUI(img)
    gi.propose()

    # Save Selection
    img.export(arg_result_path)
