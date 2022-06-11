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


class ScaleBar:

    def __init__(self):

        self.is_valid = False
        self.scale_bar_length = None
        self.scale_bar_position = None


class Image:

    valid_bar_colors = [[255, 255, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0], [128, 0, 128], [0, 128, 0]]

    def __init__(self, image: np.ndarray = None):
        self.rgb = image
        self.file_path = None
        self.file_name = None
        self.scale_bar_length = None
        self.scale_bar_position = None

    @property
    def shape(self):
        return self.rgb.shape

    def load_image(self, file_path: str):
        """Load the image."""
        self.rgb = cv2.imread(file_path)
        self.file_path = os.path.realpath(file_path)
        self.file_name = os.path.split(file_path)[-1]

    def find_scale_bar(self):

        for bc in self.valid_bar_colors:
            test = self.rgb == np.array([[bc]], dtype=self.rgb.dtype)
            test = np.all(test, axis=-1)
            test_row = np.sum(test, axis=-1)
            if np.max(test_row) < 50: continue
            max_row = np.argmax(test_row)
            indx = np.argwhere(test[max_row])[:, 0]
            shiftind = (indx + 1)
            neighb = shiftind[:-1] == indx[1:]
            total_connected = np.sum(neighb)
            if total_connected < 50: continue
            self.scale_bar_length = total_connected + 1
            self.scale_bar_position = (np.mean(indx), max_row)


class GUI:

    def __init__(self, image: Image = None):

        self.image: Image = image

        self.fig = None
        self.ax = None
        self.accept_scalebar = False
        self.image_in_fig = None

    def key_press_event(self, event):
        print('you pressed', event.key, "at", event.xdata, event.ydata)
        if event.key == "enter":
            print("Accept current grid segmentation...")
            self.fig.canvas.stop_event_loop()

        elif event.key == "m":
            self.accept_scalebar = not self.accept_scalebar
            self.ax.set_title(self._title_scalebar())
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def button_press_event(self, event):
        pass

    def _title_scalebar(self):
        if self.accept_scalebar and self.image.scale_bar_length is None:
            return "No Scale-bar available to accept."
        elif self.accept_scalebar and self.image.scale_bar_length is not None:
            return "Accepted Scale-bar with: {0} pixel".format(self.image.scale_bar_length)
        else:
            return "No accepted Scale-bar"

    def draw_scale_bar(self, scale_bar_length, scale_bar_position):
        print("Scalebar found.")
        plt.plot(np.array([- scale_bar_length / 2, scale_bar_length / 2]) + scale_bar_position[0],
                 np.array([scale_bar_position[1], scale_bar_position[1]]), color="r")
        self.ax.annotate("Scalebar found length: {0} pixel".format(scale_bar_length),
                         xy=(scale_bar_length / 2 + scale_bar_position[0], scale_bar_position[1]),
                         xytext=(50, 30), textcoords='offset points', size=20, color='r',
                         arrowprops=dict(facecolor='r', shrink=0.05),
                         horizontalalignment='left', verticalalignment='top')

    def propose(self):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.image.find_scale_bar()

        if self.image.scale_bar_length is not None:
            self.draw_scale_bar(self.image.scale_bar_length, self.image.scale_bar_position)
        # fig = plt.figure()

        self.ax.set_title(self._title_scalebar())
        self.image_in_fig = ax.imshow(self.image.rgb, cmap='hot')
        plt.ion()
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        info_cap = "".join(["Accept scale bar with 'm'."])
        plt.ylabel(info_cap, rotation='horizontal', ha='right')
        plt.show()
        fig.canvas.set_window_title(self.image.file_name)
        fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        fig.canvas.mpl_connect('button_press_event', self.button_press_event)
        fig.canvas.start_event_loop()
        fig.clear()
        plt.close(fig)
        plt.close("all")

    def export_choice(self, file_path):

        if self.accept_scalebar and self.image.scale_bar_length is not None:
            with open(os.path.join(file_path, "scale_bar_length.txt"), "w") as f:
                f.write(str(self.image.scale_bar_length))


if __name__ == "__main__":
    # Input arguments from command line.
    parser = argparse.ArgumentParser(description='Run FindScaleBar.')
    parser.add_argument("--file", required=True, help="Input filepath of image.")
    parser.add_argument("--result", required=False, help="Filepath to output folder.", default="output")
    args = vars(parser.parse_args())
    print("Input of argparse:", args)
    filepath = args["file"]
    result_path = args["result"]

    # filepath = "input/HG2A_30s.jpg"

    # Load Image
    img = Image()
    img.load_image(filepath)
    # Propose Grid
    gi = GUI(img)
    gi.propose()
    gi.export_choice(result_path)
