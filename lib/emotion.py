import sys
import traceback
import time
import datetime
import pathlib
import json
import requests
import cv2


def find_camera():
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
    else:
        print("Camera found")
    return camera_num


def analyze_emotion(image):
    url = "https://ai-api.userlocal.jp/face"
    res = requests.post(url, files={"image_data": image})
    json_encoded = res.content
    data = json.loads(json_encoded)
    return data

# for r in result:
#    print(f"""
#          年齢: {r['age']}
#          感情: {r['emotion']}
#          感情内訳： {r['emotion_detail']}
#          性別: {r['gender']}
#          顔の向き: {r['head_pose']}
#          顔の位置: {r['location']}
#          """)
