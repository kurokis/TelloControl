import sys
import traceback
import time
import datetime
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFilter
import tellopy
import av
import cv2


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
        self.log_filename = datetime.datetime.now().strftime("%y%m%d_%H%M%S")+".csv"

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
        img_filename = str(self.image_index)+".jpg"
        cv2.imwrite(str(self.image_save_dir / img_filename), image)
        self.image_index += 1

    def write_log(self, t, position, eulerdeg):
        log_path = self.log_save_dir/self.log_filename

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


class StateEstimator:
    def __init__(self):
        # Coordinate systems
        #
        # world:  +x=forward, +y=up on marker
        # marker: +x=right, +y=up on marker
        # camera: +x=right, +y=down on image
        # local: +x=forward, +y=up on image
        #
        #                       rvec,tvec
        #  local  -----> camera -----> marker -----> world
        #         <-----        <-----        <-----
        #          T_cl          T_mc          T_wm

        #### Acuro parameters ####
        self.aruco = cv2.aruco
        self.aruco_dictionary = self.aruco.getPredefinedDictionary(
            self.aruco.DICT_4X4_50)
        self.aruco_parameters = self.aruco.DetectorParameters_create()

        #### Drone states ####
        # Timestamp in seconds
        self.t = 0
        self.t0 = time.time()
        # Position relative to marker
        self.position = np.array([0, 0, 0])
        # Euler angles in intrisic z-y-x notation (yaw, pitch, roll)
        self.eulerdeg = np.array([0, 0, 0])

        # Whether a marker was visible in the most recent image
        self.marker_visible = False

        #### Camera parameters ####
        # Camera matrix
        self.cameraMatrix = np.array([[1000.,    0.,  360.],
                                      [0., 1000.,  480.],
                                      [0.,    0.,    1.]])
        # Distortion coefficients
        self.distCoeffs = np.array([[0.00000000e+000],
                                    [-6.32814106e-123],
                                    [4.34421711e-184],
                                    [3.68913910e-168],
                                    [-1.88715756e-092],
                                    [2.12305495e-153],
                                    [6.32814106e-123],
                                    [1.88715756e-092],
                                    [0.00000000e+000],
                                    [0.00000000e+000],
                                    [0.00000000e+000],
                                    [0.00000000e+000],
                                    [0.00000000e+000],
                                    [0.00000000e+000]])

        #### Marker board parameters ####
        objPoints = np.array([[[-0.1274,  0.054,  0.01023181],
                               [-0.0266,  0.054,  0.049005],
                               [-0.0266, -0.054,  0.049005],
                               [-0.1274, -0.054,  0.01023181]],
                              [[0.0266,  0.054,  0.049005],
                               [0.1274,  0.054,  0.01023181],
                               [0.1274, -0.054,  0.01023181],
                               [0.0266, -0.054,  0.049005]]]).astype(np.float32)
        ids = np.array([11, 22]).astype(np.float32)
        self.board = cv2.aruco.Board_create(
            objPoints, self.aruco_dictionary, ids)

        #### Internal variables ####
        n_keep = 100
        self.t_history = np.zeros(n_keep)
        self.position_history = np.zeros((n_keep, 3))
        self.eulerdeg_history = np.zeros((n_keep, 3))

        #### Images ####
        self.original_image = None
        self.overlay_image = None

    def update(self, image):
        # Update timestamp
        self.t = time.time() - self.t0

        # Save copy of the image
        self.original_image = image.copy()
        self.overlay_image = image.copy()

        # Detect markers
        markerCorners, markerIds, _ = self.aruco.detectMarkers(
            image, self.aruco_dictionary, parameters=self.aruco_parameters)

        # Compute rotation and translation relative to marker board
        r_cm = None
        t_cm = None
        _, rvec, tvec = self.aruco.estimatePoseBoard(
            markerCorners, markerIds, self.board, self.cameraMatrix, self.distCoeffs, None, None)

        if rvec is None:
            # Marker not visible
            self.marker_visible = False
        else:
            # Marker visible
            self.marker_visible = True

            rvec = np.squeeze(rvec)  # column vector to row vector
            tvec = np.squeeze(tvec)  # column vector to row vector

            cv2.aruco.drawAxis(
                self.overlay_image, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.1)
            r_cm = Rotation.from_rotvec(rvec)
            t_cm = tvec

            # Calculate transformation from marker to camera using inverse formula
            # Given rotation matrix R and translation vector t,
            # inverse rotation matrix = R^T
            # inverse translation vector = -R^T*t
            R_cm = r_cm.as_matrix()
            R_mc = R_cm.T
            t_mc = -np.matmul(R_cm.T, t_cm)

            # Homogeneous transformation matrix from marker to camera coordinates
            T_mc = np.block([[R_mc, t_mc[:, np.newaxis]], [0, 0, 0, 1]])

            # Homogeneous transformation matrix from camera to local coordinates
            T_cl = np.array([[0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1]])

            # Homogeneous transformation matrix from world to marker coordinates
            T_wm = np.array([[0, 0, -1, 0],
                             [-1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]])

            # Homogeneous transformation matrix from world to local coordinates
            T_wl = np.matmul(np.matmul(T_wm, T_mc), T_cl)

            # Decompose the matrix into rotation matrix and translation vector
            R_wl = T_wl[0:3, 0:3]
            t_wl = T_wl[0:3, 3]

            # Store estimated orientation
            self.position = np.array(t_wl)
            self.eulerdeg = Rotation.from_matrix(
                R_wl).as_euler('ZYX', degrees=True)

            # Draw text onto image
            cv2.putText(self.overlay_image, "yaw:{:6.1f} deg   pitch:{:6.1f} deg   roll:{:6.1f} deg".format(
                *self.eulerdeg), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(self.overlay_image, "  x:{:6.2f} m         y:{:6.2f} m        z:{:6.2f} m".format(
                *self.position), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Update history
        self.t_history = np.concatenate((self.t_history[1:], [self.t]))
        self.position_history = np.concatenate(
            (self.position_history[1:], [self.position]))
        self.eulerdeg_history = np.concatenate(
            (self.eulerdeg_history[1:], [self.eulerdeg]))


class Controller:
    def __init__(self):
        self.mode_auto = False

        # Variables for automatic control
        # Target position: x, y, z in meters
        self.target_position = np.array([-1, 0, 0])
        # Target attitude: yaw, pitch, roll in degrees
        self.target_attitude = np.array([0, 0, 0])

        # Count number of frames with no marker visible during auto mode
        self.marker_not_visible_count_during_auto = 0

    def key_handler(self, key, drone, se):
        if self.mode_auto == False:
            # Manual control
            #
            # q: quit
            # z: switch mode between manual and auto
            #
            #    w           t           i
            #  a s d         g         j k l
            #
            #  pitch      takeoff      vertical
            #  & roll    & landing     & yaw
            #
            if key == ord('t'):
                drone.takeoff()
            elif key == ord('g'):
                drone.land()
            elif key == ord('a'):
                drone.left(30)
            elif key == ord('d'):
                drone.right(30)
            elif key == ord('w'):
                drone.forward(30)
            elif key == ord('s'):
                drone.backward(30)
            elif key == ord('j'):
                drone.counter_clockwise(30)
            elif key == ord('l'):
                drone.clockwise(30)
            elif key == ord('i'):
                drone.up(30)
            elif key == ord('k'):
                drone.down(30)
            elif key == ord('z'):
                self.mode_auto = True
                # Change log level to avoid showing info
                drone.set_loglevel(drone.LOG_WARN)
            else:
                # hover
                # set internal values directly to avoid showing info
                drone.right_x = 0
                drone.right_y = 0
                drone.left_x = 0
                drone.left_y = 0
        else:
            # Automatic control

            # Failsafe: land if marker not visible for a long time
            if se.marker_visible == False:
                self.marker_not_visible_count_during_auto += 1
            if self.marker_not_visible_count_during_auto > 10:
                # Land and revert to manual mode immediately
                self.marker_not_visible_count_during_auto = 0
                drone.land()
                self.mode_auto = False
                return

            if key == ord('z'):
                # Change mode
                self.mode_auto = False
                # Change log level to show info
                drone.set_loglevel(drone.LOG_INFO)
                return
            elif key == ord('g'):
                # Land and revert to manual mode immediately
                drone.land()
                self.mode_auto = False
                return

            # Calculate position error
            delta_position = se.position - self.target_position

            k = 0.25  # (m/s)/m TODO: compute feedback gain
            scale_factor = 300.0  # unit/(m/s) TODO: estimate from log
            max_command = 30

            # x control
            dx = delta_position[0]
            speed_command = np.clip((k*abs(dx))*scale_factor, 0, max_command)
            if dx < 0:
                drone.forward(speed_command)
            elif dx > 0:
                drone.backward(speed_command)

            # y control
            dy = delta_position[1]
            speed_command = np.clip((k*abs(dy))*scale_factor, 0, max_command)
            if dy < 0:
                drone.left(speed_command)
            elif dy > 0:
                drone.right(speed_command)

            # z control
            dz = delta_position[2]
            speed_command = np.clip((k*abs(dz))*scale_factor, 0, max_command)
            if dz < 0:
                drone.up(speed_command)
            elif dz > 0:
                drone.down(speed_command)

            # yaw control
            delta_attitude = se.eulerdeg - self.target_attitude
            d_yaw = delta_attitude[0]

            k_yaw = 0.01
            scale_factor = 1.0  # unit/(deg/s) TODO: estimate from log
            yawrate_command = np.clip(
                (k_yaw*abs(d_yaw))*scale_factor, 0, max_command)
            if d_yaw < 0:
                drone.clockwise(yawrate_command)
            elif d_yaw > 0:
                drone.counter_clockwise(yawrate_command)


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
        print("moved")

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
        self.axs[0].set_xlim(min(ts), max(max(ts), min(ts)+0.01))
        self.axs[0].set_ylim(min(min(xs), min(ys), min(zs)),
                             max(max(xs), max(ys), max(zs),
                                 min(min(xs), min(ys), min(zs))+0.01))

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
        self.axs[2].set_xlim(min(ts), max(max(ts), min(ts)+0.01))
        self.axs[2].set_ylim(min(min(e1s), min(e2s), min(e3s)),
                             max(max(e1s), max(e2s), max(e3s),
                                 min(min(e1s), min(e2s), min(e3s))+0.01))

        self.fig.canvas.draw()


def main():
    # Initialize recorder
    rec = Recorder("./output_data")

    # Initialize state estimator
    se = StateEstimator()

    # Initialize controller
    controller = Controller()

    # Initialize plotter
    plotter = Plotter()

    # Create a drone instance
    drone = tellopy.Tello()

    try:
        ######## Connect with the drone ########
        drone.connect()
        drone.wait_for_connection(60.0)
        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')
        # Skip first 300 frames
        frame_skip = 300

        quit_flag = False
        first_cv2_imshow = True

        plotter.initialize_plot()
        while True:
            for frame in container.decode(video=0):
                ######## Manage process delay ########
                # Skip frames if needed
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()

                ######## Obtain image ########
                # Convert frame to BGR image
                image = cv2.cvtColor(
                    np.array(frame.to_image()), cv2.COLOR_RGB2BGR)

                ######## State estimation ########
                # Update state estimator
                se.update(image)

                ######## Control ########
                # Wait for key press (0xFF is for 64-bit support)
                key = (cv2.waitKey(1) & 0xFF)

                # Handle key outside the scope of the controller
                if key == ord('q'):
                    # End all process immediately
                    quit_flag = True
                    break
                elif key == ord('r'):
                    # Save image
                    rec.write_image(se.overlay_image)

                # Handle key inside the scope of the controller
                controller.key_handler(key, drone, se)

                ######## Show and export data ########
                # Show image
                cv2.imshow('Image', se.overlay_image)
                if first_cv2_imshow:
                    cv2.moveWindow('Image', 700, 0)

                # Update plot
                plotter.update(se)

                # Write video frame
                rec.write_video_frame(se.overlay_image)

                # Write states to log
                rec.write_log(se.t, se.position, se.eulerdeg)

                ######## Manage process delay ########
                # Calculate number of frames to skip
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)

            if quit_flag == True:
                for _ in range(8):
                    drone.land()
                    time.sleep(1)
                break

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
        for _ in range(8):
            drone.land()
            time.sleep(1)
    finally:
        rec.release()
        drone.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
