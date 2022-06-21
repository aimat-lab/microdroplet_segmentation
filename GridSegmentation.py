import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from configs import load_config
from image import Image
from grid import Grid

mpl.use("Qt5Cairo")
mpl.rcParams["keymap.back"] = ['backspace']
mpl.rcParams["keymap.forward"] = []
mpl.rcParams["keymap.save"] = ['ctrl+s']  # Remove s here to be able to use w,s,d,a as arrows
mpl.rcParams["keymap.home"] = ['h', 'home']
mpl.rcParams["keymap.xscale"] = ['ctrl+k', 'ctrl+L']
mpl.rcParams["keymap.yscale"] = ['ctrl+l']
mpl.rcParams["keymap.all_axes"] = ['ctrl+a']  # deprecated


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
            self.grid.propose_grid(self.image.data())
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
            self.image_in_fig = self.ax.imshow(self.image.data, cmap='hot', vmax=self.bright)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif event.key == "+":
            if self.bright - self.brightness_increase < 0:
                return
            self.bright -= self.brightness_increase
            self.image_in_fig.remove()
            self.image_in_fig = self.ax.imshow(self.image.data, cmap='hot', vmax=self.bright)
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
        self.image_in_fig = plt.imshow(self.image.data, cmap='hot', vmax=self.bright)
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
    img.load(arg_file_path)
    img_gray = img.convert("GRAY").astype("float").rescale_intensity()

    # Make Grid
    grd = Grid()
    grd.propose_grid(img_gray.data)

    # Propose Grid
    gi = GUI(img_gray, grd)
    gi.run()

    # Export choice
    grd.save(arg_result_path)
