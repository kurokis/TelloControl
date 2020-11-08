import sys
import cv2
from PIL import Image, ImageDraw, ImageFilter
import pathlib
from numpy import array_repr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


######## Parameters ########

save_dir = pathlib.Path("./output_data/markers")

# Create save directory if it does not exist
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)


# Marker board definition
#
# top view
#
#          base
#    --------*<center-
#    \               /
#     \             /
#      \           /
#       \         /
#    left\       /right
#         \     /
#          \   /
#           \ /
#
# front view
#    -------- --------
#   |  left  | right  |
#   |        |        |
#   |        *<center |
#   |        |        |
#   |        |        |
#    -------- --------
#

base_length = 0.308
left_length = 0.165
right_length = 0.165
base_height = 0.254

left_face = {
    "11": {
        "center_x": 0.0825,  # meters, measure from upper left of left face
        "center_y": 0.127,  # meters, measure from upper left of left face
        "size": 0.108  # meters
    },
}

right_face = {

    "22": {
        "center_x": 0.0825,  # meters, measure from upper left of right face
        "center_y": 0.127,  # meters, measure from upper left of right face
        "size": 0.108  # meters
    },
}

apex_angle = np.arccos((left_length**2 + right_length**2 -
                        base_length**2) / (2 * left_length * right_length))
left_angle = np.arccos((base_length**2 + left_length**2 -
                        right_length**2) / (2 * base_length * left_length))
right_angle = np.arccos((base_length**2 + right_length**2 -
                         left_length**2) / (2 * base_length * right_length))
print("apex angle", apex_angle * 180 / np.pi)
print("left angle", left_angle * 180 / np.pi)
print("right angle", right_angle * 180 / np.pi)

# Calculate position
objPoints = []
for key in left_face.keys():
    marker_id = int(key)
    x = left_face[key]["center_x"]
    y = left_face[key]["center_y"]
    size = left_face[key]["size"]
    upper_left_x = (x - 0.5 * size) * np.cos(left_angle) - 0.5 * base_length
    upper_left_y = y + 0.5 * size - 0.5 * base_height
    upper_left_z = (x - 0.5 * size) * np.sin(left_angle)

    upper_right_x = (x + 0.5 * size) * np.cos(left_angle) - 0.5 * base_length
    upper_right_y = y + 0.5 * size - 0.5 * base_height
    upper_right_z = (x + 0.5 * size) * np.sin(left_angle)

    lower_right_x = (x + 0.5 * size) * np.cos(left_angle) - 0.5 * base_length
    lower_right_y = y - 0.5 * size - 0.5 * base_height
    lower_right_z = (x + 0.5 * size) * np.sin(left_angle)

    lower_left_x = (x - 0.5 * size) * np.cos(left_angle) - 0.5 * base_length
    lower_left_y = y - 0.5 * size - 0.5 * base_height
    lower_left_z = (x - 0.5 * size) * np.sin(left_angle)

    objPoints.append(
        [[upper_left_x, upper_left_y, upper_left_z],
         [upper_right_x, upper_right_y, upper_right_z],
         [lower_right_x, lower_right_y, lower_right_z],
         [lower_left_x, lower_left_y, lower_left_z]])

for key in right_face.keys():
    marker_id = int(key)
    x = right_face[key]["center_x"]
    y = right_face[key]["center_y"]
    size = right_face[key]["size"]
    upper_left_x = 0.5 * base_length - \
        (right_length - (x - 0.5 * size)) * np.cos(right_angle)
    upper_left_y = y + 0.5 * size - 0.5 * base_height
    upper_left_z = (right_length - (x - 0.5 * size)) * np.sin(right_angle)

    upper_right_x = 0.5 * base_length - \
        (right_length - (x + 0.5 * size)) * np.cos(right_angle)
    upper_right_y = y + 0.5 * size - 0.5 * base_height
    upper_right_z = (right_length - (x + 0.5 * size)) * np.sin(right_angle)

    lower_right_x = 0.5 * base_length - \
        (right_length - (x + 0.5 * size)) * np.cos(right_angle)
    lower_right_y = y - 0.5 * size - 0.5 * base_height
    lower_right_z = (right_length - (x + 0.5 * size)) * np.sin(right_angle)

    lower_left_x = 0.5 * base_length - \
        (right_length - (x - 0.5 * size)) * np.cos(right_angle)
    lower_left_y = y - 0.5 * size - 0.5 * base_height
    lower_left_z = (right_length - (x - 0.5 * size)) * np.sin(right_angle)

    objPoints.append(
        [[upper_left_x, upper_left_y, upper_left_z],
         [upper_right_x, upper_right_y, upper_right_z],
         [lower_right_x, lower_right_y, lower_right_z],
         [lower_left_x, lower_left_y, lower_left_z]])

# objPoints must be a float32 object
objPoints = np.array(objPoints).astype(np.float32)

# Save results
np.set_printoptions(threshold=sys.maxsize)
with open(str(save_dir / "marker_board.txt"), 'w') as f:
    f.write(array_repr(objPoints))
print("marker board saved to {}".format(str(save_dir)))

# Plot results
fig = plt.figure(figsize=plt.figaspect(1))
ax = Axes3D(fig)
for marker in objPoints:
    ax.plot(marker[:, 0], marker[:, 1], marker[:, 2],
            "o-", color="#00aa00", ms=4, mew=0.5)
ax.plot([-base_length / 2, base_length / 2, base_length / 2, -base_length / 2, -base_length / 2],
        [base_height / 2, base_height / 2, -base_height /
            2, -base_height / 2, base_height / 2],
        [0, 0, 0, 0, 0])
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_zlim(-0.3, 0.3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()


# Create images for printing
#
#
#    ---------->x (width)
#    |
#    |   A4 paper
#    |
#    y (height)
#
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# Create left face image
base_image = Image.new('RGB', (2100, 2970), (255, 255, 255))
for key in left_face.keys():
    marker_id = int(key)
    x = left_face[key]["center_x"]
    y = left_face[key]["center_y"]
    size = left_face[key]["size"]
    marker = aruco.drawMarker(dictionary, marker_id, int(10000 * size))
    marker_image = Image.fromarray(marker)
    base_image.paste(
        marker_image, (int(10000 * (x - size / 2)), int(10000 * (y - size / 2))))
    cv2.imwrite(str(save_dir / (str(marker_id) + ".png")), marker)
base_image.save(str(save_dir / 'marker_board_left.png'))

# Create right face image

base_image = Image.new('RGB', (2100, 2970), (255, 255, 255))
for key in right_face.keys():
    marker_id = int(key)
    x = right_face[key]["center_x"]
    y = right_face[key]["center_y"]
    size = right_face[key]["size"]
    marker = aruco.drawMarker(dictionary, marker_id, int(10000 * size))
    marker_image = Image.fromarray(marker)
    base_image.paste(
        marker_image, (int(10000 * (x - size / 2)), int(10000 * (y - size / 2))))
    cv2.imwrite(str(save_dir / (str(marker_id) + ".png")), marker)
base_image.save(str(save_dir / 'marker_board_right.png'))


###################################################
######## Pose estimation with marker board ########

objPoints = np.array([[[0, 0, 0], [5, 0, 0], [5, -5, 0], [0, -5, 0]],
                      [[7, 0, 0], [9, 0, 0], [9, -2, 0], [7, -2, 0]]]).astype(np.float32)

ids = np.array([0, 1]).astype(np.float32)

board = cv2.aruco.Board_create(objPoints, dictionary, ids)

image = cv2.imread("./input_data/3.jpg")


# Camera matrix
cameraMatrix = np.array([[1000., 0., 360.],
                         [0., 1000., 480.],
                         [0., 0., 1.]])
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

# Detect markers
markerCorners, markerIds, _ = aruco.detectMarkers(
    image, dictionary, parameters=parameters)


retval, rvec, tvec = aruco.estimatePoseBoard(
    markerCorners, markerIds, board, cameraMatrix, distCoeffs, None, None)

cv2.aruco.drawAxis(
    image, cameraMatrix, distCoeffs, rvec, tvec, 1)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
# plt.show()
#
# squaresX = 4
# squaresY = 3
# squareLength = 0.036
# markerLength = squareLength*0.8
#
######### Create charuco board ########
#
# aruco = cv2.aruco
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# board = aruco.CharucoBoard_create(
#    squaresX, squaresY, squareLength, markerLength, aruco_dict)
#
# print("board", board.chessboardCorners)
#
# base_image.save(str(save_dir / 'marker_map.png'))
# print("marker map saved to {}".format(str(save_dir)))
#


# cap = cv2.VideoCapture("output_data/save/z_flipping/output.avi")
# video_writer = cv2.VideoWriter("output_data/save/z_flipping/output_board_estimation.avi", cv2.VideoWriter_fourcc(
#    'M', 'J', 'P', 'G'), 10, (960, 720))
# while(cap.isOpened()):
#    ret, frame = cap.read()
#
#    image = frame.copy()
#    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#    markerCorners, markerIds, _ = aruco.detectMarkers(
#        frame, dictionary, parameters=parameters)
#
#    retval, rvec, tvec = aruco.estimatePoseBoard(
#        markerCorners, markerIds, board, cameraMatrix, distCoeffs, None, None)
#
#    if rvec is not None:
#        cv2.aruco.drawAxis(
#            image, cameraMatrix, distCoeffs, rvec, tvec, 1)
#
#    cv2.imshow("Image", image)
#
#    video_writer.write(image)
#
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#
# cap.release()
# cv2.destroyAllWindows()
