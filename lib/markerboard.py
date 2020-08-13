import sys
import cv2
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from numpy import array_repr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')


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

def board_v2_parameters():
    base_height = 0.254
    base_length = 0.308
    left_length = 0.165
    right_length = 0.165

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

    params = {
        "base_height": base_height,
        "base_length": base_length,
        "left_length": left_length,
        "right_length": right_length,
        "left_face": left_face,
        "right_face": right_face,
    }

    return params


def aruco_parameters():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_param = cv2.aruco.DetectorParameters_create()
    return aruco_dict, aruco_param


def sample_camera_parameters():
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
    return cameraMatrix, distCoeffs


def create_marker_board():
    ######### Aruco board ########
    # Axis direction of objPoints: +x=right, +y=up
    # Order of objPoints: upper-left, upper-right, lower right, lower left
    # See the docs for details
    # https://docs.opencv.org/master/db/da9/tutorial_aruco_board_detection.html

    ######### Board v1 ########
    # objPoints = np.array([[[0, 0, 0], [5, 0, 0], [5, -5, 0], [0, -5, 0]],
    #                      [[7, 0, 0], [9, 0, 0], [9, -2, 0], [7, -2, 0]]]).astype(np.float32)
    # ids = np.array([0, 1]).astype(np.float32)

    ######### Board v2 ########
    params = board_v2_parameters()
    base_length = params["base_length"]
    base_height = params["base_height"]
    left_length = params["left_length"]
    right_length = params["right_length"]
    left_face = params["left_face"]
    right_face = params["right_face"]

    apex_angle = np.arccos((left_length**2+right_length**2 -
                            base_length**2)/(2*left_length*right_length))
    left_angle = np.arccos((base_length**2+left_length**2 -
                            right_length**2)/(2*base_length*left_length))
    right_angle = np.arccos((base_length**2+right_length**2 -
                             left_length**2)/(2*base_length*right_length))
    # print("apex angle", apex_angle*180/np.pi)
    # print("left angle", left_angle*180/np.pi)
    # print("right angle", right_angle*180/np.pi)

    objPoints = []
    ids = []
    for key in left_face.keys():
        marker_id = int(key)
        x = left_face[key]["center_x"]
        y = left_face[key]["center_y"]
        size = left_face[key]["size"]
        upper_left_x = (x-0.5*size)*np.cos(left_angle)-0.5*base_length
        upper_left_y = y+0.5*size-0.5*base_height
        upper_left_z = (x-0.5*size)*np.sin(left_angle)

        upper_right_x = (x+0.5*size)*np.cos(left_angle)-0.5*base_length
        upper_right_y = y+0.5*size-0.5*base_height
        upper_right_z = (x+0.5*size)*np.sin(left_angle)

        lower_right_x = (x+0.5*size)*np.cos(left_angle)-0.5*base_length
        lower_right_y = y-0.5*size-0.5*base_height
        lower_right_z = (x+0.5*size)*np.sin(left_angle)

        lower_left_x = (x-0.5*size)*np.cos(left_angle)-0.5*base_length
        lower_left_y = y-0.5*size-0.5*base_height
        lower_left_z = (x-0.5*size)*np.sin(left_angle)

        objPoints.append(
            [[upper_left_x, upper_left_y, upper_left_z],
             [upper_right_x, upper_right_y, upper_right_z],
             [lower_right_x, lower_right_y, lower_right_z],
             [lower_left_x, lower_left_y, lower_left_z]])
        ids.append(marker_id)

    for key in right_face.keys():
        marker_id = int(key)
        x = right_face[key]["center_x"]
        y = right_face[key]["center_y"]
        size = right_face[key]["size"]
        upper_left_x = 0.5*base_length - \
            (right_length-(x-0.5*size))*np.cos(right_angle)
        upper_left_y = y+0.5*size-0.5*base_height
        upper_left_z = (right_length-(x-0.5*size))*np.sin(right_angle)

        upper_right_x = 0.5*base_length - \
            (right_length-(x+0.5*size))*np.cos(right_angle)
        upper_right_y = y+0.5*size-0.5*base_height
        upper_right_z = (right_length-(x+0.5*size))*np.sin(right_angle)

        lower_right_x = 0.5*base_length - \
            (right_length-(x+0.5*size))*np.cos(right_angle)
        lower_right_y = y-0.5*size-0.5*base_height
        lower_right_z = (right_length-(x+0.5*size))*np.sin(right_angle)

        lower_left_x = 0.5*base_length - \
            (right_length-(x-0.5*size))*np.cos(right_angle)
        lower_left_y = y-0.5*size-0.5*base_height
        lower_left_z = (right_length-(x-0.5*size))*np.sin(right_angle)

        objPoints.append(
            [[upper_left_x, upper_left_y, upper_left_z],
             [upper_right_x, upper_right_y, upper_right_z],
             [lower_right_x, lower_right_y, lower_right_z],
             [lower_left_x, lower_left_y, lower_left_z]])
        ids.append(marker_id)

    # aruco expects float32 for objPoints and ids
    objPoints = np.array(objPoints).astype(np.float32)
    ids = np.array(ids).astype(np.float32)

    # Load aruco parameters
    aruco_dict, _ = aruco_parameters()

    # Create board object
    board = cv2.aruco.Board_create(objPoints, aruco_dict, ids)

    # Write results to a text file
    # p.set_printoptions(threshold=sys.maxsize)
    # save_dir = Path("./output_data")
    # save_dir.mkdir(parents=True, exist_ok=True)
    # with open(str(save_dir/"marker_board.txt"), 'w') as f:
    #     f.write(array_repr(board.objPoints))
    # print("marker board saved to {}".format(str(save_dir)))

    return board


def visualize_marker_board(board):

    # Load parameters
    params = board_v2_parameters()
    base_length = params["base_length"]
    base_height = params["base_height"]

    # Create figure
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = Axes3D(fig)

    # Draw objPoints
    for marker in board.objPoints:
        ax.plot(marker[:, 0], marker[:, 1], marker[:, 2],
                "o-", color="#00aa00", ms=4, mew=0.5)

    # Draw base board
    ax.plot([-base_length/2, base_length/2, base_length/2, -base_length/2, -base_length/2],
            [base_height/2, base_height/2, -base_height /
                2, -base_height/2, base_height/2],
            [0, 0, 0, 0, 0])

    # Plot settings
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def write_marker_board_v1_to_image(save_dir):
    # Create save directory if it does not exist
    save_dir.mkdir(parents=True, exist_ok=True)

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

    # Initialize aruco
    aruco_dict, _ = aruco_parameters()

    # Create base image
    base_image = Image.new('RGB', (width, height), (255, 255, 255))

    for key in marker_map.keys():
        # Marker parameters
        marker_id = int(key)
        x = marker_map[key]["x"]
        y = marker_map[key]["y"]
        size = marker_map[key]["size"]

        # Create single marker
        marker = cv2.aruco.drawMarker(aruco_dict, marker_id, size)

        # Paste marker onto base image
        marker_image = Image.fromarray(marker)
        base_image.paste(marker_image, (x, y))

    # Save image
    base_image.save(str(save_dir/'marker_board_v1.png'))
    print("marker board saved to {}".format(str(save_dir)))


def write_marker_board_v2_to_image(save_dir):
    # Create images for printing
    #
    #
    #    ---------->x (width)
    #    |
    #    |   A4 paper
    #    |
    #    y (height)
    #

    # Load parameters
    params = board_v2_parameters()
    left_face = params["left_face"]
    right_face = params["right_face"]

    # Create aruco dictionary
    aruco_dict, _ = aruco_parameters()

    # Create save directory if it does not exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Create left face image
    base_image = Image.new('RGB', (2100, 2970), (255, 255, 255))
    for key in left_face.keys():
        marker_id = int(key)
        x = left_face[key]["center_x"]
        y = left_face[key]["center_y"]
        size = left_face[key]["size"]
        marker = cv2.aruco.drawMarker(aruco_dict, marker_id, int(10000*size))
        marker_image = Image.fromarray(marker)
        base_image.paste(
            marker_image, (int(10000*(x-size/2)), int(10000*(y-size/2))))
    base_image.save(str(save_dir/'marker_board_v2_left.png'))

    # Create right face image
    base_image = Image.new('RGB', (2100, 2970), (255, 255, 255))
    for key in right_face.keys():
        marker_id = int(key)
        x = right_face[key]["center_x"]
        y = right_face[key]["center_y"]
        size = right_face[key]["size"]
        marker = cv2.aruco.drawMarker(aruco_dict, marker_id, int(10000*size))
        marker_image = Image.fromarray(marker)
        base_image.paste(
            marker_image, (int(10000*(x-size/2)), int(10000*(y-size/2))))
    base_image.save(str(save_dir/'marker_board_v2_right.png'))

    print("marker board saved to {}".format(str(save_dir)))


def pose_estimation_from_image(image_path):
    # Load camera parameters
    cameraMatrix, distCoeffs = sample_camera_parameters()

    # Load aruco parameters
    aruco_dict, aruco_param = aruco_parameters()

    # Create marker board
    board = create_marker_board()

    # Read image
    image = cv2.imread(str(image_path))

    # Detect markers
    markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
        image, aruco_dict, parameters=aruco_param)

    # Estimate pose from aruco board
    _, rvec, tvec = cv2.aruco.estimatePoseBoard(
        markerCorners, markerIds, board, cameraMatrix, distCoeffs, None, None)

    # Draw axis if pose estimation successful
    if rvec is not None:
        cv2.aruco.drawAxis(
            image, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

    # Show image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def pose_estimation_from_video(video_path, output_path):
    # Load camera parameters
    cameraMatrix, distCoeffs = sample_camera_parameters()

    # Load aruco parameters
    aruco_dict, aruco_param = aruco_parameters()

    # Create marker board
    board = create_marker_board()

    cap = cv2.VideoCapture(str(video_path))

    video_writer = cv2.VideoWriter(str(output_path),
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (960, 720))

    frame_number = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            print("done")
            break
        else:
            image = frame.copy()

            # Detect markers
            markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
                frame, aruco_dict, parameters=aruco_param)

            # Estimate pose from aruco board
            _, rvec, tvec = cv2.aruco.estimatePoseBoard(
                markerCorners, markerIds, board, cameraMatrix, distCoeffs, None, None)

            # Draw axis if pose estimation successful
            if rvec is not None:
                cv2.aruco.drawAxis(
                    image, cameraMatrix, distCoeffs, rvec, tvec, 0.1)

            # Show image
            cv2.imshow("Image", image)

            # Write video
            video_writer.write(image)

            # Key handler
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Increment frame number
            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Create a marker board
    board = create_marker_board()

    # Write the marker board v1 to image
    # write_marker_board_v1_to_image(Path("../output_data/markers"))

    # Write the marker board v2 to image
    write_marker_board_v2_to_image(Path("../output_data/markers"))

    # Visualize marker board in 3D
    visualize_marker_board(board)

    # Estimate pose from image
    pose_estimation_from_image(Path("../input_data/sample/board.jpg"))

    # Estimate pose from video
    output_dir = Path("../output_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_estimation_from_video(
        Path("../input_data/sample/board.avi"), output_dir/"board_out.avi")
