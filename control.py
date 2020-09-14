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


def control_thread(queue):
    # Initialize recorder
    rec = Recorder("./output_data")

    # Initialize state estimator
    se = StateEstimator()

    # Initialize controller
    controller = Controller()

    # Initialize plotter
    plotter = Plotter()

    # Initialize face recognition
    fr = None

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

        """ADD"""
        # target
        target_list = np.array(
            [[-1.2, 0.3, 0.0], [-0.9, 0.0, 0.0], [-1.2, -0.3, 0.0], [-1.5, 0.0, 0.0]])
        n_target = len(target_list)
        target_counter = 0
        target_judge = 0.2
        """ADD END"""

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

                ######## Update face recognition status ########
                while not queue.empty():
                    fr = queue.get(block=False)
                if fr is not None:
                    if len(fr) == 0:
                        pass
                    else:
                        r = fr[0]
                        emotion = r['emotion']
                        print("emotion:", emotion)

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

                """ADD"""
                # Define target
                target_counter = target_counter % n_target
                print(target_counter)
                controller.target_position = target_list[target_counter]

                # Judge target
                delta_target = se.position - controller.target_position
                delta_target_norm = np.linalg.norm(delta_target)
                if delta_target_norm < target_judge:
                    target_counter += 1
                    # print('next target')
                """ADD END"""

                # Handle key inside the scope of the controller
                controller.key_handler(key, drone, se)

                ######## Show and export data ########
                # Show image
                cv2.imshow('Image', se.overlay_image)
                if first_cv2_imshow:
                    first_cv2_imshow = False
                    cv2.moveWindow('Image', 300, 0)

                # Update plot
                plotter.update(se)

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


def face_recognition_thread(queue):
    # Parameters
    device_num = 0
    output_dir = pathlib.Path("output_data")
    file_prefix = "camera_capture_cycle"
    ext = "jpg"
    cycle = 300
    window_name = "Face Recognition"

    # Open webcam
    cap = cv2.VideoCapture(device_num)

    if not cap.isOpened():
        print("webcam not opened")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        else:
            n = (n + 1) % cycle

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if n == 0:
                filename = '{}_{}.{}'.format(
                    file_prefix, datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'), ext)
                image_path = str(output_dir / filename)
                cv2.imwrite(image_path, frame)

                # Face recognition
                image = open(image_path, 'rb').read()
                url = "https://ai-api.userlocal.jp/face"
                res = requests.post(url, files={"image_data": image})
                data = json.loads(res.content)
                result = data['result']
                # for r in result:
                #    print(f"""
                #          年齢: {r['age']}
                #          感情: {r['emotion']}
                #          感情内訳： {r['emotion_detail']}
                #          性別: {r['gender']}
                #          顔の向き: {r['head_pose']}
                #          顔の位置: {r['location']}
                #          """)
                queue.put(result)

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    queue = queue.Queue()

    t1 = threading.Thread(target=control_thread, args=(queue,))
    t2 = threading.Thread(target=face_recognition_thread, args=(queue,))
    t2.setDaemon(True)
    t1.start()
    t2.start()
