import cv2
from PIL import Image, ImageDraw, ImageFilter
import pathlib

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
width = 1200
height = 1200

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


#### Functions ####

def generate_aruco_marker(marker_id, size):
    # id: integer between 0 and 50
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    marker = aruco.drawMarker(dictionary, marker_id, size)

    cv2.imwrite(str(save_dir / (str(marker_id)+".png")), marker)

    return marker


# Create and save marker map

base_image = Image.new('RGB', (width, height), (255, 255, 255))

for key in marker_map.keys():
    marker_id = int(key)
    x = marker_map[key]["x"]
    y = marker_map[key]["y"]
    size = marker_map[key]["size"]

    marker = generate_aruco_marker(marker_id, size)

    marker_image = Image.fromarray(marker)
    base_image.paste(marker_image, (x, y))

base_image.save(str(save_dir / 'marker_map.png'))
print("marker map saved to {}".format(str(save_dir)))
