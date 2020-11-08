import numpy
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib

######## Parameters ########

save_dir = pathlib.Path("./output_data/charuco_board")

# Create save directory if it does not exist
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

squaresX = 7
squaresY = 5
squareLength = 0.036
markerLength = squareLength * 0.8

######## Create charuco board ########

aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard_create(
    squaresX, squaresY, squareLength, markerLength, aruco_dict)
imboard = board.draw((2000, 2000))

cv2.imwrite(str(save_dir / "chessboard.tiff"), imboard)
print("calibration board saved to {}".format(str(save_dir)))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(imboard, cmap=mpl.cm.gray, interpolation="nearest")
ax.axis("off")
plt.show()
