import sys
import traceback
import time
import datetime
import pathlib
import json
import requests
import queue
import threading
import cv2

import socket


def server_thread(queue):
    # Start server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 8080))
    s.listen(5)

    while True:
        clientsocket, address = s.accept()

        print(f"Connection from {address} has been established!")

        json_encoded = None
        while not queue.empty():
            json_encoded = queue.get(block=False)

        if json_encoded is not None:
            clientsocket.send(json_encoded)
        else:
            clientsocket.send(b'{"result":[],"status":"ng"}\n')
        clientsocket.close()


def face_recognition_thread(queue):
    # Parameters
    output_dir = pathlib.Path("output_data")
    file_prefix = "camera_capture_cycle"
    ext = "jpg"
    cycle = 30 * 20
    window_name = "Face Recognition"

    camera_num = -1
    for camera_number in range(0, 10):
        cap = cv2.VideoCapture(camera_number)
        ret, frame = cap.read()
        if ret:
            frame_rate = cap.get(5)
            print("Camera number:{}¥t frame rate:{}".format(
                camera_number, frame_rate))
            if 29 < frame_rate < 31:
                camera_num = camera_number
                break

    if camera_num == -1:
        print("Could not find camera")
        return
    else:
        print("Camera found")

    # Open webcam
    cap = cv2.VideoCapture(camera_num)

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
                json_encoded = res.content
                queue.put(json_encoded)

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    queue = queue.Queue()

    t1 = threading.Thread(target=face_recognition_thread, args=(queue,))
    t2 = threading.Thread(target=server_thread, args=(queue,))
    t2.setDaemon(True)
    t1.start()
    t2.start()

    # for r in result:
    #    print(f"""
    #          年齢: {r['age']}
    #          感情: {r['emotion']}
    #          感情内訳： {r['emotion_detail']}
    #          性別: {r['gender']}
    #          顔の向き: {r['head_pose']}
    #          顔の位置: {r['location']}
    #          """)
