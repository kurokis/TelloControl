from numpy import array_repr
import numpy as np
import cv2
import os
import pathlib

######## Parameters ########

calib_img_dir = pathlib.Path("./input_data/calibration_images")

save_dir = pathlib.Path("./output_data/calibration_results")

# Create save directory if it does not exist
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

squaresX = 7
squaresY = 5
squareLength = 0.036
markerLength = squareLength * 0.8

aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard_create(
    squaresX, squaresY, squareLength, markerLength, aruco_dict)


image_paths = np.array([str(calib_img_dir / f)
                        for f in os.listdir(calib_img_dir) if f.endswith(".jpg")])


def read_chessboards(image_paths):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in image_paths:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize


def calibrate_camera(allCorners, allIds, imsize):
    """
    Calibrates the camera using the detected corners.
    """

    cameraMatrixInit = np.array([[1000., 0., imsize[0] / 2.],
                                 [0., 1000., imsize[1] / 2.],
                                 [0., 0., 1.]])

    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
             cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


allCorners, allIds, imsize = read_chessboards(image_paths)
ret, cameraMatrix, distCoeffs, rvecs, tvecs = calibrate_camera(
    allCorners, allIds, imsize)


with open(str(save_dir / "camera_matrix.txt"), 'w') as f:
    f.write(array_repr(cameraMatrix))

with open(str(save_dir / "distortion_coefficients.txt"), 'w') as f:
    f.write(array_repr(distCoeffs))

print("calibration results saved to {}".format(str(save_dir)))
