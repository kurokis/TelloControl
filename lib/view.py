import pathlib
import datetime
import numpy as np
import cv2
import matplotlib
"""manage error mentioned below on mac"""
"""AttributeError: 'FigureManagerMac' object has no attribute 'window'"""
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Recorder:
    def __init__(self, output_dir, frame_width=960, frame_height=720):
        # Variables
        self.image_index = 0
        self.output_dir = pathlib.Path(output_dir)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_save_dir = self.output_dir / "videos"
        self.image_save_dir = self.output_dir / "images"
        self.log_save_dir = self.output_dir / "logs"
        self.log_filename = datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".csv"

        # Create directory for saving image
        pathlib.Path(self.image_save_dir).mkdir(parents=True, exist_ok=True)

        # Create directory for saving video
        pathlib.Path(self.video_save_dir).mkdir(parents=True, exist_ok=True)

        # Create directory for saving video
        pathlib.Path(self.log_save_dir).mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        self.video_writer = cv2.VideoWriter(str(self.video_save_dir / "output.avi"), cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (self.frame_width, self.frame_height))

    def write_video_frame(self, image):
        # Write video frame
        self.video_writer.write(image)

    def write_image(self, image):
        img_filename = str(self.image_index) + ".jpg"
        cv2.imwrite(str(self.image_save_dir / img_filename), image)
        self.image_index += 1

    def write_log(self, t, position, eulerdeg):
        log_path = self.log_save_dir / self.log_filename

        # Write header if log file does not yet exist
        if not log_path.exists():
            with open(log_path, mode='w') as f:
                f.write("t(s),x(m),y(m),z(m),yaw(deg),pitch(deg),roll(deg)\n")

        # Append row
        with open(log_path, mode='a') as f:
            s = ""
            data = [t] + list(position) + list(eulerdeg)
            for d in data[:-1]:
                s += str(d) + ","
            s += str(data[-1]) + "\n"
            f.write(s)

    def release(self):
        self.video_writer.release()


class Plotter():
    def __init__(self):
        figs, axs = plt.subplots(3)
        self.fig = figs
        self.axs = axs

        # Dummy plot
        self.p_tx, = self.axs[0].plot(np.zeros(1), np.zeros(1), label='x')
        self.p_ty, = self.axs[0].plot(np.zeros(1), np.zeros(1), label='y')
        self.p_tz, = self.axs[0].plot(np.zeros(1), np.zeros(1), label='z')
        self.p_yz, = self.axs[1].plot(np.zeros(1), np.zeros(1), label='y-z')
        self.p_te1, = self.axs[2].plot(
            np.zeros(1), np.zeros(1), label='yaw')
        self.p_te2, = self.axs[2].plot(
            np.zeros(1), np.zeros(1), label='pitch')
        self.p_te3, = self.axs[2].plot(
            np.zeros(1), np.zeros(1), label='roll')

        axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper left')
        axs[2].legend(loc='upper left')

        self.move_figure(self.fig, 0, 0)

    def move_figure(self, f, x, y):
        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'WXAgg':
            f.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            f.canvas.manager.window.move(x, y)

    def initialize_plot(self):
        # Set matplotlib to non-blocking mode
        plt.show(block=False)

    def update(self, se):
        ts = se.t_history
        xs = se.position_history[:, 0]
        ys = se.position_history[:, 1]
        zs = se.position_history[:, 2]
        e1s = se.eulerdeg_history[:, 0]
        e2s = se.eulerdeg_history[:, 1]
        e3s = se.eulerdeg_history[:, 2]

        # time history of x (depth)
        self.p_tx.set_xdata(ts)
        self.p_tx.set_ydata(xs)
        self.p_ty.set_xdata(ts)
        self.p_ty.set_ydata(ys)
        self.p_tz.set_xdata(ts)
        self.p_tz.set_ydata(zs)
        self.axs[0].set_xlim(min(ts), max(max(ts), min(ts) + 0.01))
        self.axs[0].set_ylim(min(min(xs), min(ys), min(zs)),
                             max(max(xs), max(ys), max(zs),
                                 min(min(xs), min(ys), min(zs)) + 0.01))

        # y-z position
        n_plot = min(len(ys), 10)
        self.p_yz.set_xdata(-ys[-n_plot:-1])
        self.p_yz.set_ydata(zs[-n_plot:-1])
        self.axs[1].set_xlim(-1, 1)
        self.axs[1].set_ylim(-1, 1)

        # time history of euler angles
        self.p_te1.set_xdata(ts)
        self.p_te1.set_ydata(e1s)
        self.p_te2.set_xdata(ts)
        self.p_te2.set_ydata(e2s)
        self.p_te3.set_xdata(ts)
        self.p_te3.set_ydata(e3s)
        self.axs[2].set_xlim(min(ts), max(max(ts), min(ts) + 0.01))
        self.axs[2].set_ylim(min(min(e1s), min(e2s), min(e3s)),
                             max(max(e1s), max(e2s), max(e3s),
                                 min(min(e1s), min(e2s), min(e3s)) + 0.01))

        self.fig.canvas.draw()
