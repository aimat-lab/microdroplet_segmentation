import cv2
import numpy as np
import os
import scipy as sp
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

# import yaml

mpl.use("Qt5Cairo")
mpl.rcParams["keymap.back"] = ['backspace']
mpl.rcParams["keymap.forward"] = []
mpl.rcParams["keymap.save"] = ['ctrl+s']  # Remove s here to be able to use w,s,d,a as arrows
mpl.rcParams["keymap.home"] = ['h', 'home']


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
        self.file_path = os.path.realpath(file_path)


class Grid:
    min_expected_segments = 3
    max_possible_segments = 1000
    default_origin = np.array([0, 0], dtype="int")

    def __init__(self, image: Image = None):
        self.image = image
        self.image_intensity = self.image.gray_norm
        self.origin = self.default_origin  # is y,x
        self.delta_x = 10.0
        self.delta_y = 10.0
        self.num_x = np.array([0, 4], dtype="int")
        self.num_y = np.array([0, 4], dtype="int")

    def make_grid(self):
        grid_x_pos = np.arange(self.num_x[0], self.num_x[1]) * self.delta_x + self.origin[1]
        grid_y_pos = np.arange(self.num_y[0], self.num_y[1]) * self.delta_y + self.origin[0]
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
        self.origin = np.array([np.amin(grid_y_pos), np.amin(grid_x_pos)])
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
        xgrid = np.sort(np.array(self.grid_x_pos))
        ygrid = np.sort(np.array(self.grid_y_pos))
        # index_map_x = np.arange(len(xgrid), dtype="int")
        # index_map_y = np.arange(len(ygrid), dtype="int")
        xgrid = xgrid[np.logical_and(xgrid > 0, xgrid < self.image.shape[1])]
        ygrid = ygrid[np.logical_and(ygrid > 0, ygrid < self.image.shape[0])]

        def padd_grid(grid, delta, maxlen):
            if grid[0] - delta > 0:
                return padd_grid(np.concatenate([np.array([grid[0]]) - delta, grid]), delta, maxlen)
            elif grid[-1] + delta < maxlen:
                return padd_grid(np.concatenate([grid, np.array([grid[-1]]) + delta]), delta, maxlen)
            return grid

        self.grid_y_pos = padd_grid(ygrid, self.estimated_delta_y, self.image.shape[0])  # .tolist()
        self.grid_x_pos = padd_grid(xgrid, self.estimated_delta_x, self.image.shape[1])  # .tolist()

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
        """Main function to propose the grid for segmentation."""
        dx, dy = self.find_grid_spacing_fft()
        self.origin = np.array([0, 0])
        self.delta_x = dx
        self.delta_y = dy
        self.num_x = np.array([0, int(self.image_intensity.shape[1] / dx)], dtype="int")
        self.num_y = np.array([0, int(self.image_intensity.shape[0] / dy)], dtype="int")
        self.optimize_grid()
        return self.make_grid()

    def export_grid(self, filepath):
        np.save(os.path.join(filepath, "grid_x.npy"), self.grid_x_pos)
        np.save(os.path.join(filepath, "grid_y.npy"), self.grid_y_pos)


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
        self.selected_grid_line_x = None
        self.selected_grid_line_y = None
        self.interactive_mode = False
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
            is_locked = "Lock active" if self.interactive_mode else ""
            self.ax.set_title(is_locked)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif event.key == "n":
            self.grid.propose_grid()
            self.grid.make_grid()
            self.set_grid_x(flush=False)
            self.set_grid_y(flush=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif event.key == "r":
            pass

        elif event.key == "u":
            pass

        elif event.key == "-":
            if self.bright + self.brightness_increase > 1: return
            self.bright += self.brightness_increase
            self.image_in_fig.remove()
            self.image_in_fig = self.ax.imshow(self.image.gray_norm, cmap='hot', vmax=self.bright)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif event.key == "+":
            if self.bright - self.brightness_increase < 0: return
            self.bright -= self.brightness_increase
            self.image_in_fig.remove()
            self.image_in_fig = self.ax.imshow(self.image.gray_norm, cmap='hot', vmax=self.bright)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def button_press_event(self, event):
        """Mouse click event analysis for line selection."""
        print('Click event at', event.xdata, event.ydata)
        if not event.inaxes: return
        if not self.interactive_mode: return
        if event.inaxes != self.ax: return
        xpos = event.xdata
        ypos = event.ydata
        diff_x = np.abs(np.array(self.grid.grid_x_pos) - xpos)
        diff_y = np.abs(np.array(self.grid.grid_y_pos) - ypos)
        foundx = np.any(diff_x < self.pixel_box)
        foundy = np.any(diff_y < self.pixel_box)
        if foundx:
            print("Selected x line")
            idx = np.argmin(diff_x)
            if self.selected_grid_line_x is not None:
                self.interactive_deselect_line_x()
            self.interactive_select_line_x(idx)
        elif foundy:
            print("Selected y line")
            idx = np.argmin(diff_y)
            if self.selected_grid_line_y is not None:
                self.interactive_deselect_line_y()
            self.interactive_select_line_y(idx)

    def interactive_select_line_x(self, idx, flush=True):
        hl = self.fig_x_lines[idx]
        if self.use_blit:
            hl.set_color('y')
            self.draw_grid()
        else:
            hl.remove()
            self.fig_x_lines[idx] = self.ax.axvline(x=self.grid_x_pos[idx], color='y', linestyle='-')
            if flush:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        self.selected_grid_line_x = idx

    def interactive_select_line_y(self, idx, flush=True):
        hl = self.fig_y_lines[idx]
        if self.use_blit:
            hl.set_color('y')
            self.draw_grid()
        else:
            hl.remove()
            self.fig_y_lines[idx] = self.ax.axhline(y=self.grid_y_pos[idx], color='y', linestyle='-')
            if flush:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        self.selected_grid_line_y = idx

    def set_grid_x(self, flush=True):
        """Redraw c lines."""
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
        """Draw grid with animated position, i.e. stretched and moved."""
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
        else:
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
        # plt.xlim([0, self.image.shape[1]])
        # plt.ylim([self.image.shape[0], 0])
        is_locked = "Lock active" if self.interactive_mode else ""
        self.title = plt.title(is_locked)
        info = "".join(["Press 'm' to toggle lock and click line.\n",
                        "Press 'w', 'a', 's', 'd' to stretch.\n",
                        "Press 'left', 'up', ... key to move.\n",
                        "Press '+', '-' to increase brightness (view only).\n",
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
    img.compute_gray_image()
    img.compute_slices()

    # Make Grid
    grd = Grid(image=img)
    grd.propose_grid()

    # Propose Grid
    gi = GUI(img, grd)
    gi.run()

    # Export choice
    grd.export_grid(arg_result_path)
