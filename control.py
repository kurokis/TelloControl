import sys
import traceback
import time
import datetime
import pathlib
import json
import requests
import queue
import threading
import numpy as np
import cv2
import av
import tellopy
from lib.model import StateEstimator, Controller
from lib.view import Recorder, Plotter
import socket
import json


def control_thread():
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
    
    # emotion
    emotion_detail = None

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

        #"""ADD"""
        # target
        #target_list = np.array(
        #    [[-1.2, 0.3, 0.0], [-0.9, 0.0, 0.0], [-1.2, -0.3, 0.0], [-1.5, 0.0, 0.0]])
        #n_target = len(target_list)
        #target_counter = 0
        #target_judge = 0.2
        #"""ADD END"""

        plotter.initialize_plot()
        while True:
            ######## Update face recognition status ########
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('127.0.0.1', 8080))
                    s.settimeout(3)
                    s.sendall(b'')
                    json_encoded = s.recv(1024)

                    data = json.loads(json_encoded)
                    if len(data['result']) > 0:
                        #emotion = data['result'][0]['emotion']
                        emotion_detail = data['result'][0]['emotion_detail']
                        #print("emotion:", emotion)
            except Exception:
                pass

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

                #"""ADD"""
                ## Define target
                #target_counter = target_counter % n_target
                ## print(target_counter)
                #controller.target_position = target_list[target_counter]
                #
                ## Judge target
                #delta_target = se.position - controller.target_position
                #delta_target_norm = np.linalg.norm(delta_target)
                #if delta_target_norm < target_judge:
                #    target_counter += 1
                #    # print('next target')
                #"""ADD END"""

                # Handle key inside the scope of the controller
                controller.key_handler(key, drone, se)

                ######## Show and export data ########
                # Show image
                cv2.imshow('Image', se.overlay_image)
                if first_cv2_imshow:
                    first_cv2_imshow = False
                    cv2.moveWindow('Image', 300, 0)

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
