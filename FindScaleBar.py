import cv2
import numpy as np
import os
import argparse
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Qt5Cairo")
mpl.rcParams["keymap.back"] = ['backspace']
mpl.rcParams["keymap.forward"] = []


class ScaleBar:

    valid_bar_colors = [[255, 255, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0], [128, 0, 128], [0, 128, 0]]

    def __init__(self):
        self.image_reference = None
        self.user_accepted = None
        self.is_valid = False
        self.length = None
        self.position = None

    def locate(self, rgb: np.ndarray, image_reference: str = None):
        self.image_reference = image_reference
        for bc in self.valid_bar_colors:
            test = rgb == np.array([[bc]], dtype=rgb.dtype)
            test = np.all(test, axis=-1)
            test_row = np.sum(test, axis=-1)
            if np.max(test_row) < 50:
                continue
            max_row = np.argmax(test_row)
            index = np.argwhere(test[max_row])[:, 0]
            shift_index = (index + 1)
            neighbour = shift_index[:-1] == index[1:]
            total_connected = np.sum(neighbour)
            if total_connected < 50:
                continue
            self.length = total_connected + 1
            self.position = (np.mean(index), max_row)
            self.is_valid = True

    @staticmethod
    def save_yaml_file(out, file_name):
        with open(file_name, 'w') as yaml_file:
            yaml.dump(out, yaml_file, default_flow_style=False)

    def export(self, file_path):
        self.save_yaml_file({"image_reference": self.image_reference,
                             "is_valid": self.is_valid,
                             "length": int(self.length),
                             "position": [int(x) for x in self.position],
                             "user_accepted": self.user_accepted},
                            os.path.join(file_path, "ScaleBar.yaml"))


class Image:

    def __init__(self, image: np.ndarray = None):
        self.rgb = image
        self.file_path = None
        self.scale_bar = ScaleBar()

    @property
    def shape(self):
        return self.rgb.shape

    @property
    def file_name(self):
        return os.path.splitext(os.path.basename(self.file_path))[0]

    @property
    def file_extension(self):
        return os.path.splitext(os.path.basename(self.file_path))[1]

    def load_image(self, file_path: str):
        """Load the image."""
        self.rgb = cv2.imread(file_path)
        self.file_path = os.path.realpath(file_path)

    def find_scale_bar(self):
        self.scale_bar.locate(self.rgb, self.file_path)


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
        if self.accept_scalebar and not self.image.scale_bar.is_valid:
            return "No Scale-bar available to accept."
        elif self.accept_scalebar and self.image.scale_bar.is_valid:
            return "Accepted Scale-bar with: {0} pixel".format(self.image.scale_bar.length)
        else:
            return "No accepted Scale-bar"

    def draw_scale_bar(self, scale_bar: ScaleBar):
        scale_bar_length = scale_bar.length
        scale_bar_position = scale_bar.position
        print("Scalebar found.")
        plt.plot(np.array([- scale_bar_length / 2, scale_bar_length / 2]) + scale_bar_position[0],
                 np.array([scale_bar_position[1], scale_bar_position[1]]), color="r")
        self.ax.annotate("Scalebar found length: {0} pixel".format(scale_bar_length),
                         xy=(scale_bar_length / 2 + scale_bar_position[0], scale_bar_position[1]),
                         xytext=(50, 30), textcoords='offset points', size=20, color='r',
                         arrowprops=dict(facecolor='r', shrink=0.05),
                         horizontalalignment='left', verticalalignment='top')

    def propose(self):
        # fig = plt.figure()
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        if self.image.scale_bar.is_valid:
            self.draw_scale_bar(self.image.scale_bar)

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

    def accept_choice(self, file_path):

        if self.accept_scalebar and self.image.scale_bar.is_valid:
            self.image.scale_bar.user_accepted = self.accept_scalebar
            self.image.scale_bar.export(file_path)


if __name__ == "__main__":
    # Input arguments from command line.
    parser = argparse.ArgumentParser(description='Run FindScaleBar.')
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
    img.find_scale_bar()

    # Propose Grid
    gi = GUI(img)
    gi.propose()
    gi.accept_choice(arg_result_path)
