import sys
import traceback
import tellopy
import av
import cv2
import time
import pathlib
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from scipy.spatial.transform import Rotation


class Recorder:
    def __init__(self, output_dir, frame_width=960, frame_height=720):
        # Variables
        self.image_index = 0
        self.output_dir = pathlib.Path(output_dir)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_save_dir = self.output_dir / "videos"
        self.image_save_dir = self.output_dir / "images"

        # Create directory for saving image
        pathlib.Path(self.image_save_dir).mkdir(parents=True, exist_ok=True)

        # Create directory for saving video
        pathlib.Path(self.video_save_dir).mkdir(parents=True, exist_ok=True)

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

        #### Drone states ####
        # Position relative to marker
        self.position = np.array([0, 0, 0])
        # Euler angles in intrisic z-y-x notation (yaw, pitch, roll)
        self.eulerdeg = np.array([0, 0, 0])

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

        #### Marker map parameters ####
        self.marker_map = {
            "0": {"x": 100, "y": 100, "size": 500},
            "1": {"x": 700, "y": 100, "size": 200},
            "2": {"x": 700, "y": 400, "size": 200},
            "3": {"x": 100, "y": 700, "size": 200},
            "4": {"x": 400, "y": 700, "size": 200},
            "5": {"x": 700, "y": 700, "size": 200},
        }

        # Scale factor to convert marker size to real world units
        # Define by meters/pixel
        self.scale_factor = 0.108/500

        #### Images ####
        self.original_image = None
        self.overlay_image = None

        #### Initialization ####
        self.aruco = cv2.aruco
        self.aruco_dictionary = self.aruco.getPredefinedDictionary(
            self.aruco.DICT_4X4_50)
        self.aruco_parameters = self.aruco.DetectorParameters_create()

    def update(self, image):
        # Update state from image

        # Save original image
        self.original_image = image.copy()

        # Detect markers
        markerCorners, markerIds, _ = self.aruco.detectMarkers(
            image, self.aruco_dictionary, parameters=self.aruco_parameters)

        # Compute rotation and translation relative to marker
        r_cm = None
        t_cm = None
        if markerIds is not None:
            for corners, markerId in zip(markerCorners, markerIds):
                try:
                    marker_size = self.marker_map[str(markerId[0])]['size']
                except KeyError:
                    # Skip if a marker not defined in marker map is found
                    continue

                rvecs, tvecs, _objPoints = self.aruco.estimatePoseSingleMarkers(
                    corners, marker_size*self.scale_factor, self.cameraMatrix, self.distCoeffs)

                # Remove unnecessary axis
                tvec = np.squeeze(tvecs)
                rvec = np.squeeze(rvecs)

                cv2.aruco.drawAxis(
                    image, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.15/2)

                if markerId[0] == 0:
                    r_cm = Rotation.from_rotvec(rvec)
                    t_cm = tvec

        if r_cm is not None:
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

            self.position = np.array(t_wl)
            self.eulerdeg = Rotation.from_matrix(
                R_wl).as_euler('ZYX', degrees=True)

            cv2.putText(image, "yaw:{:6.1f} deg   pitch:{:6.1f} deg   roll:{:6.1f} deg".format(
                *self.eulerdeg), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "  x:{:6.2f} m         y:{:6.2f} m        z:{:6.2f} m".format(
                *self.position), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        self.overlay_image = image


def main():
    # Initialize recorder
    rec = Recorder("./output_data")

    # Initialize state estimator
    se = StateEstimator()

    # Create a drone instance
    drone = tellopy.Tello()

    try:
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

        # skip first 300 frames
        frame_skip = 300

        while True:
            quit_flag = None
            for frame in container.decode(video=0):
                # Skip frames if needed
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()

                # Convert frame to BGR image
                image = cv2.cvtColor(
                    np.array(frame.to_image()), cv2.COLOR_RGB2BGR)

                # Update state estimator
                se.update(image)

                # Show image
                cv2.imshow('Image', se.overlay_image)

                # Write video frame
                rec.write_video_frame(se.overlay_image)

                # Wait for key press (0xFF is for 64-bit support)
                key = (cv2.waitKey(1) & 0xFF)
                if key == ord('q'):
                    quit_flag = True
                    break
                elif key == ord('t'):
                    drone.takeoff()
                elif key == ord('g'):
                    drone.land()
                elif key == ord('a'):
                    drone.left(10)
                elif key == ord('d'):
                    drone.right(10)
                elif key == ord('w'):
                    drone.forward(10)
                elif key == ord('s'):
                    drone.backward(10)
                elif key == ord('j'):
                    drone.counter_clockwise(10)
                elif key == ord('l'):
                    drone.clockwise(10)
                elif key == ord('i'):
                    drone.up(10)
                elif key == ord('k'):
                    drone.down(10)
                elif key == ord('x'):
                    # stop
                    drone.left(0)
                    drone.forward(0)
                    drone.counter_clockwise(0)
                    drone.up(0)
                elif key == ord('r'):
                    rec.write_image(se.overlay_image)

                # Calculate number of frames to skip
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)

            if quit_flag == True:
                break

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        rec.release()
        drone.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
