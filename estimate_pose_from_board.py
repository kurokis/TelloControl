import cv2
from PIL import Image, ImageDraw, ImageFilter
import pathlib
import numpy as np
import matplotlib.pyplot as plt

######## Parameters ########

save_dir = pathlib.Path("./output_data/markers")

# Create save directory if it does not exist
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

# Base image
#
#    ---------->x (width)
#    |
#    |
#    |
#    y (height)
#
width = 1000
height = 1000

# Marker map
# x: x position of upper left corner
# y: y position of upper left corner
marker_map = {
    "0": {"x": 100, "y": 100, "size": 500},
    "1": {"x": 700, "y": 100, "size": 200},
    "2": {"x": 700, "y": 400, "size": 200},
    "3": {"x": 100, "y": 700, "size": 200},
    "4": {"x": 400, "y": 700, "size": 200},
    "5": {"x": 700, "y": 700, "size": 200},
}

# Camera matrix
cameraMatrix = np.array([[1000.,    0.,  360.],
                         [0., 1000.,  480.],
                         [0.,    0.,    1.]])
# Distortion coefficients
distCoeffs = np.array([[0.00000000e+000],
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


#### Functions ####
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# Original marker map
# objPoints = np.array([[[0, 0, 0], [5, 0, 0], [5, -5, 0], [0, -5, 0]],
#                      [[7, 0, 0], [9, 0, 0], [9, -2, 0], [7, -2, 0]]]).astype(np.float32)

objPoints = np.array([[[-0.1274,  0.054,  0.01023181],
                       [-0.0266,  0.054,  0.049005],
                       [-0.0266, -0.054,  0.049005],
                       [-0.1274, -0.054,  0.01023181]],
                      [[0.0266,  0.054,  0.049005],
                       [0.1274,  0.054,  0.01023181],
                       [0.1274, -0.054,  0.01023181],
                       [0.0266, -0.054,  0.049005]]]).astype(np.float32)


ids = np.array([11, 22]).astype(np.float32)

board = cv2.aruco.Board_create(objPoints, dictionary, ids)


# plt.imshow(marker, cmap='gray')
image = cv2.imread("./input_data/3.jpg")

# Detect markers
markerCorners, markerIds, _ = aruco.detectMarkers(
    image, dictionary, parameters=parameters)

retval, rvec, tvec = aruco.estimatePoseBoard(
    markerCorners, markerIds, board, cameraMatrix, distCoeffs, None, None)

if rvec is not None:
    cv2.aruco.drawAxis(
        image, cameraMatrix, distCoeffs, rvec, tvec, 1)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
# plt.show()


cap = cv2.VideoCapture("input_data/board_test.avi")
video_writer = cv2.VideoWriter("output_data/board_test_results.avi", cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (960, 720))
while(cap.isOpened()):
    ret, frame = cap.read()
    print("new frame")

    image = frame.copy()

    markerCorners, markerIds, _ = aruco.detectMarkers(
        frame, dictionary, parameters=parameters)

    retval, rvec, tvec = aruco.estimatePoseBoard(
        markerCorners, markerIds, board, cameraMatrix, distCoeffs, None, None)

    if rvec is not None:
        cv2.aruco.drawAxis(
            image, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

    cv2.imshow("Image", image)

    video_writer.write(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
