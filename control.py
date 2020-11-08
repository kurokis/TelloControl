import sys
import traceback
import time
import datetime
import pathlib
import json
import requests
#import queue
#import threading
import numpy as np
import cv2
import av
import tellopy
from lib.model import StateEstimator, Controller
from lib.view import Recorder, Plotter
from lib.emotion import find_camera, analyze_emotion
#import socket
from concurrent.futures import ThreadPoolExecutor


def control_thread():
    # Initialize recorder
    rec = Recorder("./output_data")

    # Initialize state estimator
    se = StateEstimator()

    # Initialize controller
    controller = Controller()

    # Initialize plotter
    plotter = Plotter()

    # Open webcam
    camera_found = False
    camera_num = find_camera()
    if camera_num != -1:
        camera_found = True
        cap = cv2.VideoCapture(camera_num)

    # Subprocess executor
    executor = ThreadPoolExecutor()

    # Create a drone instance
    drone = tellopy.Tello()

    # emotion
    emotion_process_running = False
    t_last_call = time.time()
    emotion_process_interval = 5.0
    emotion_detail = None

    try:
        ######## Connect with the drone ########
        drone.connect()
        drone.wait_for_connection(30.0)
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
    except Exception as ex:
        pass

    try:
        quit_flag = False
        first_cv2_imshow = True

        plotter.initialize_plot()
        while True:
            ######## Update face recognition status ########
            if camera_found:
                # Pass image data to emotion API if emotion process is not busy
                if (emotion_process_running == False) & (time.time() - t_last_call > emotion_process_interval):
                    ret, webcam_frame = cap.read()
                    if not ret:
                        continue
                    cv2.imwrite("latest.jpg", webcam_frame)
                    image = open("latest.jpg", 'rb').read()
                    future = executor.submit(analyze_emotion, image)
                    emotion_process_running = True
                    t_last_call = time.time()

                # Store information into emotion_detail when API returns data
                if emotion_process_running:
                    if future.done():
                        data = future.result()
                        if len(data['result']) > 0:
                            emotion_detail = data['result'][0]['emotion_detail']
                        emotion_process_running = False
                        cv2.imshow('Webcam', webcam_frame)

            ######## Process video stream from Tello ########
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
                cv2.imshow('Tello', se.overlay_image)
                if first_cv2_imshow:
                    first_cv2_imshow = False
                    cv2.moveWindow('Tello', 300, 0)

                # Update plot
                plotter.update(se, emotion_detail)

                # Write video frame
                rec.write_video_frame(se.overlay_image)

                # Write states to log
                rec.write_log(se.t, se.position, se.eulerdeg)

                ######## Manage process delay ########
                # Calculate number of frames to skip
                if frame.time_base < 1.0 / 60:
                    time_base = 1.0 / 60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time) / time_base)

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
    control_thread()
