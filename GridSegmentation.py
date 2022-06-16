import cv2
import numpy as np
import os
import scipy as sp
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import yaml

mpl.use("Qt5Cairo")
mpl.rcParams["keymap.back"] = ['backspace']
mpl.rcParams["keymap.forward"] = []
mpl.rcParams["keymap.save"] = ['ctrl+s']  # Remove s here to be able to use w,s,d,a as arrows
mpl.rcParams["keymap.home"] = ['h', 'home']
mpl.rcParams["keymap.xscale"] = ['ctrl+k', 'ctrl+L']
mpl.rcParams["keymap.yscale"] = ['ctrl+l']
mpl.rcParams["keymap.all_axes"] = ['ctrl+a']  # deprecated


class Image:

    def __init__(self, image=None):
        self.file_path = None
        self.rgb = image
        self.gray_norm = None
        self.slice_x = None
        self.slice_y = None

    def compute_gray_image(self):
        if self.rgb is not None:
            gray = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)
            self.gray_norm = gray / np.amax(gray)  # *min(np.mean(self.gray), 255)

    def compute_slices(self):
        if self.gray_norm is not None:
            self.slice_x = np.sum(self.gray_norm, axis=0)
            self.slice_y = np.sum(self.gray_norm, axis=1)

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
        self.rgb = cv2.imread(file_path)
        self.file_path = os.path.normpath(file_path)


class Grid:

    min_expected_segments = 3
    max_possible_segments = 1000
    default_origin = np.array([0, 0], dtype="int")  # is [y, x]
    default_delta_x = 50
    default_delta_y = 50
    default_num_x = np.array([0, 4], dtype="int")  # is [left, right] >= 0
    default_num_y = np.array([0, 4], dtype="int")  # is [left, right] >= 0

    def __init__(self, image: Image):
        self.image = image
        self.image_intensity = self.image.gray_norm
        self.file_path = None
        self.origin = self.default_origin  # is [y, x]
        self.delta_x = self.default_delta_x
        self.delta_y = self.default_delta_y
        self.num_x = self.default_num_x  # is [left, right] >= 0
        self.num_y = self.default_num_y  # is [left, right] >= 0

    @staticmethod
    def save_yaml_file(out, file_name):
        with open(file_name, 'w') as yaml_file:
            yaml.dump(out, yaml_file, default_flow_style=False)

    def make_grid(self):
        grid_x_pos = np.arange(-self.num_x[0], self.num_x[1]) * self.delta_x + self.origin[1]
        grid_y_pos = np.arange(-self.num_y[0], self.num_y[1]) * self.delta_y + self.origin[0]
        return grid_x_pos, grid_y_pos

    def optimize_grid(self):

        def opt_axis(gp: np.ndarray, delta, sl: np.ndarray, axis: int):
            ra = np.arange(delta)
            shift = np.expand_dims(ra, axis=-1) + np.expand_dims(gp, axis=0) - np.amin(gp)
            shift[shift >= self.image.shape[axis]] = 0
            shift = np.array(shift, dtype="int")
            # print(np.argmin(np.sum(self.slice_x[sx],axis=-1)))
            offset = ra[np.argmin(np.sum(sl[shift], axis=-1))]
            gp = gp + offset
            return gp

        grid_x_pos, grid_y_pos = self.make_grid()
        grid_x_pos = opt_axis(grid_x_pos, self.delta_x, self.image.slice_x, 1)
        grid_y_pos = opt_axis(grid_y_pos, self.delta_y, self.image.slice_y, 0)
        self.origin = np.array([grid_y_pos[self.num_y[0]], grid_x_pos[self.num_x[0]]])
        return grid_x_pos, grid_y_pos

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

    def find_grid_spacing_fft(self):
        """Estimate spacing by main frequency component in x,y-slices."""
        image = self.image_intensity
        min_expected_segments = self.min_expected_segments
        max_possible_segments = self.max_possible_segments
        # Main frequency for each direction
        sl_x = np.sum(image, axis=0)
        sl_y = np.sum(image, axis=1)
        fy = self._get_fft_main_frequency_1d(
            sl_y, 1.0 / image.shape[0] * min_expected_segments, 1 / image.shape[0] * max_possible_segments)
        fx = self._get_fft_main_frequency_1d(
            sl_x, 1.0 / image.shape[1] * min_expected_segments, 1 / image.shape[1] * max_possible_segments)
        return 1 / fx, 1 / fy

    def find_peak_position_slices(self,
                                  find_kwargs_x=None,
                                  find_kwargs_y=None,
                                  distance_tolerance=0.75):
        image = self.image_intensity
        find_kwargs_x = {} if find_kwargs_x is None else find_kwargs_x
        find_kwargs_y = {} if find_kwargs_y is None else find_kwargs_y
        sl_x = np.sum(image, axis=0)
        sl_y = np.sum(image, axis=1)
        x_peaks, _ = sp.signal.find_peaks(sl_x, **find_kwargs_x)
        y_peaks, _ = sp.signal.find_peaks(sl_y, **find_kwargs_y)
        return x_peaks, y_peaks

    def propose_grid(self):
        """Main function to run the grid for segmentation."""
        dx, dy = self.find_grid_spacing_fft()
        self.origin = np.array([0, 0])
        self.delta_x = dx
        self.delta_y = dy
        self.num_x = np.array([0, int(self.image_intensity.shape[1] / dx)], dtype="int")
        self.num_y = np.array([0, int(self.image_intensity.shape[0] / dy)], dtype="int")
        self.optimize_grid()
        return self.make_grid()

    def make_xy_grid_array(self):
        grid_x_pos, grid_y_pos = self.make_grid()
        num_x = len(grid_x_pos)
        num_y = len(grid_y_pos)
        grid_x_pos = np.repeat(np.expand_dims(np.expand_dims(grid_x_pos, axis=-1), axis=1), num_y, axis=1)
        grid_y_pos = np.repeat(np.expand_dims(np.expand_dims(grid_y_pos, axis=-1), axis=0), num_x, axis=0)
        return np.concatenate([grid_x_pos, grid_y_pos], axis=-1)

    def save(self, directory_path, file_name="GridProperties.yaml"):
        self.file_path = os.path.normpath(os.path.join(directory_path, file_name))
        self.save_yaml_file({"image_reference": self.image.file_path if self.image is not None else None,
                             "grid_reference": str(self.file_path),
                             "origin": [int(x) for x in self.origin],
                             "delta_x": float(self.delta_x),
                             "delta_y": float(self.delta_y),
                             "num_x": [int(x) for x in self.num_x],
                             "num_y": [int(x) for x in self.num_y]},
                            self.file_path)
        np.save(os.path.join(directory_path, "grid.npy"), self.make_xy_grid_array())

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


class GUI:

    pixel_box = 10
    offset_step_x = 5
    stretch_step_x = 1
    offset_step_y = 5
    stretch_step_y = 1
    default_size = 10
    brightness_increase = 0.1
    use_blit = True

    def __init__(self, image: Image, grid: Grid):
        self.image = image
        self.grid = grid

        self.fig_x_lines = []
        self.fig_y_lines = []

        self.fig = None
        self.ax = None
        self.image_in_fig = None
        self.interactive_mode = True
        self.add_grid_segments = 1
        self.title = None
        self.bright = 1.0
        self.background = None
        self.canvas = None

    def key_press_event(self, event):
        """Keyboard click event."""
        # print('you pressed', event.key, "at", event.xdata, event.ydata)
        if event.key == "enter":
            print("Accept current grid segmentation...")
            self.fig.canvas.stop_event_loop()
        # w, a, s, d
        elif event.key == "d":
            self.grid.apply_stretch(dx=self.stretch_step_x)
            self.draw_grid()
        elif event.key == "s":
            self.grid.apply_stretch(dy=self.stretch_step_y)
            self.draw_grid()
        elif event.key == "a":
            self.grid.apply_stretch(dx=- self.stretch_step_x)
            self.draw_grid()
        elif event.key == "w":
            self.grid.apply_stretch(dy=- self.stretch_step_y)
            self.draw_grid()
        # j, i, k, l
        elif event.key == "j":
            self.grid.add_grid_column(nx=np.array([1, 0], dtype="int")*self.add_grid_segments)
            self.set_grid_x()
        elif event.key == "l":
            self.grid.add_grid_column(nx=np.array([0, 1], dtype="int")*self.add_grid_segments)
            self.set_grid_x()
        elif event.key == "i":
            self.grid.add_grid_row(ny=np.array([1, 0], dtype="int") * self.add_grid_segments)
            self.set_grid_y()
        elif event.key == "k":
            self.grid.add_grid_row(ny=np.array([0, 1], dtype="int") * self.add_grid_segments)
            self.set_grid_y()
        # left, right, up, down
        elif event.key == "right":
            self.grid.apply_offset(x=self.offset_step_x)
            self.draw_grid()
        elif event.key == "down":
            self.grid.apply_offset(y=self.offset_step_y)
            self.draw_grid()
        elif event.key == "left":
            self.grid.apply_offset(x=-self.offset_step_x)
            self.draw_grid()
        elif event.key == "up":
            self.grid.apply_offset(y=-self.offset_step_y)
            self.draw_grid()

        elif event.key == "m":
            print("Toggle interactive...")
            self.interactive_mode = not self.interactive_mode
            self.set_title()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif event.key == "u":
            print("Change grid adding/removing...")
            self.add_grid_segments = self.add_grid_segments * (-1)
            self.set_title()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif event.key == "n":
            self.grid.reset_default_grid()
            self.grid.make_grid()
            self.set_grid_x(flush=False)
            self.set_grid_y(flush=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif event.key == "r":
            self.grid.propose_grid()
            self.grid.make_grid()
            self.set_grid_x(flush=False)
            self.set_grid_y(flush=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif event.key == "-":
            if self.bright + self.brightness_increase > 1:
                return
            self.bright += self.brightness_increase
            self.image_in_fig.remove()
            self.image_in_fig = self.ax.imshow(self.image.gray_norm, cmap='hot', vmax=self.bright)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif event.key == "+":
            if self.bright - self.brightness_increase < 0:
                return
            self.bright -= self.brightness_increase
            self.image_in_fig.remove()
            self.image_in_fig = self.ax.imshow(self.image.gray_norm, cmap='hot', vmax=self.bright)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def button_press_event(self, event):
        """Mouse click event analysis for line selection."""
        print('Click event at', event.xdata, event.ydata)
        if not event.inaxes:
            return
        if not self.interactive_mode:
            return
        if event.inaxes != self.ax:
            return
        grid_x_pos, grid_y_pos = self.grid.make_grid()
        diff_x = np.abs(np.array(grid_x_pos) - event.xdata)
        diff_y = np.abs(np.array(grid_y_pos) - event.ydata)
        found_x = np.any(diff_x < self.pixel_box)
        found_y = np.any(diff_y < self.pixel_box)
        if found_x:
            idx = np.argmin(diff_x)
            self.grid.shift_origin(nx=idx)
            self.set_grid_x()
        elif found_y:
            idx = np.argmin(diff_y)
            self.grid.shift_origin(ny=idx)
            self.set_grid_y()

    def set_grid_x(self, flush=True):
        """Redraw x lines."""
        for hl in self.fig_x_lines:
            hl.remove()
        grid_x_pos, grid_y_pos = self.grid.make_grid()
        colors = ["tab:pink"] * len(grid_x_pos)
        colors[self.grid.num_x[0]] = "y"
        new_lines = []
        for i, j in enumerate(grid_x_pos):
            vis = False if self.use_blit else True
            ly = self.ax.plot([j, j], [grid_y_pos[0], grid_y_pos[-1]],
                              color=colors[i], linestyle='-', visible=vis, animated=self.use_blit)[0]
            new_lines.append(ly)
        self.fig_x_lines = new_lines
        if flush:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def set_grid_y(self, flush=True):
        """Redraw y lines."""
        for hl in self.fig_y_lines:
            hl.remove()
        grid_x_pos, grid_y_pos = self.grid.make_grid()
        colors = ["tab:pink"] * len(grid_y_pos)
        colors[self.grid.num_y[0]] = "y"
        new_lines = []
        for i, j in enumerate(grid_y_pos):
            vis = False if self.use_blit else True
            ly = self.ax.plot([grid_x_pos[0], grid_x_pos[-1]], [j, j],
                              color=colors[i], linestyle='-', visible=vis, animated=self.use_blit)[0]
            new_lines.append(ly)
        self.fig_y_lines = new_lines
        if flush:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def draw_grid(self, restore=True):
        """Draw grid with animated position, i.e. stretched and moved. Much faster!!"""
        if self.use_blit:
            if restore:
                self.fig.canvas.restore_region(self.background)
            grid_x_pos, grid_y_pos = self.grid.make_grid()
            for i, hl in enumerate(self.fig_x_lines):
                hl.set_xdata((grid_x_pos[i], grid_x_pos[i]))
                hl.set_ydata((grid_y_pos[0], grid_y_pos[-1]))
                hl.set_visible(True)
                self.ax.draw_artist(hl)
            for i, hl in enumerate(self.fig_y_lines):
                grid_y_pos = self.grid.make_grid()[1]
                hl.set_xdata((grid_x_pos[0], grid_x_pos[-1]))
                hl.set_ydata((grid_y_pos[i], grid_y_pos[i]))
                hl.set_visible(True)
                self.ax.draw_artist(hl)
            self.fig.canvas.blit()
            return
        # we need to set grid again here
        self.set_grid_y(flush=False)
        self.set_grid_x(flush=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def set_lines_visible(self, vis=True):
        for hl in self.fig_y_lines:
            hl.set_visible(vis)
        for vl in self.fig_x_lines:
            vl.set_visible(vis)

    def draw_event(self, event):
        if self.use_blit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
            self.draw_grid(restore=False)

    def set_title(self):
        is_locked = not self.interactive_mode
        is_add = "Add" if self.add_grid_segments > 0 else "Remove"
        title = "Mode: %s , Lock active: %s" % (is_add, is_locked)
        self.ax.set_title(title)

    def run(self):
        plt.ion()
        fig, ax = plt.subplots()
        self.fig = fig
        self.canvas = fig.canvas
        self.ax = ax
        self.use_blit = self.use_blit and self.canvas.supports_blit
        print("Use blit:", self.use_blit)
        self.image_in_fig = plt.imshow(self.image.gray_norm, cmap='hot', vmax=self.bright)
        self.set_grid_y(flush=False)
        self.set_grid_x(flush=False)
        # Don't need to set axis limit.
        # plt.xlim([0, self.image.shape[1]])
        # plt.ylim([self.image.shape[0], 0])
        self.set_title()
        info = "".join(["Press 'm' to lock origin.\n",
                        "Press 'w', 'a', 's', 'd' to stretch.\n",
                        "Press 'left', 'up', 'down', 'right' key to move.\n",
                        "Press 'j', 'i', 'k', 'l', to add segments.\n",
                        "Press 'u' change between add and remove segments.\n",
                        "Press '+', '-' to change brightness (view only).\n",
                        "Press 'n' make default grid.\n",
                        "Press 'r' reset to auto grid."])
        plt.ylabel(info, rotation='horizontal', ha='right')
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        fig_manager.set_window_title(self.image.file_name)
        plt.show()
        fig.canvas.mpl_connect('draw_event', self.draw_event)
        fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        fig.canvas.mpl_connect('button_press_event', self.button_press_event)
        fig.canvas.start_event_loop()
        fig.clear()
        plt.close(fig)
        plt.close("all")


if __name__ == "__main__":
    # Input arguments from command line.
    parser = argparse.ArgumentParser(description='Run FindScaleBar.')
    parser.add_argument("--file", required=True, help="Input filepath of image.")
    args = vars(parser.parse_args())
    print("Input of argparse:", args)

    # File and path information
    arg_file_path = args["file"]
    # arg_file_path = "output/HG2A_30s/HG2A_30s_select.jpg"
    arg_result_path = os.path.dirname(arg_file_path)
    arg_file_name = os.path.basename(arg_file_path)

    # Load Image
    img = Image()
    img.load_image(arg_file_path)
    img.compute_gray_image()
    img.compute_slices()

    # Make Grid
    grd = Grid(image=img)
    grd.propose_grid()

    # Propose Grid
    gi = GUI(img, grd)
    gi.run()

    # Export choice
    grd.save(arg_result_path)
