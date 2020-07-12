import sys
import traceback
import tellopy
import av
import cv2
import numpy
import time
import pathlib
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from scipy.spatial.transform import Rotation


marker_map = {
    "0": {"x": 100, "y": 100, "size": 500},
    "1": {"x": 700, "y": 100, "size": 200},
    "2": {"x": 700, "y": 400, "size": 200},
    "3": {"x": 100, "y": 700, "size": 200},
    "4": {"x": 400, "y": 700, "size": 200},
    "5": {"x": 700, "y": 700, "size": 200},
}

scale_factor = 0.112/500

cameraMatrix = np.array([[1000.,    0.,  360.],
    [   0., 1000.,  480.],
    [   0.,    0.,    1.]])
distCoeffs = np.array([[ 0.00000000e+000],
    [-6.32814106e-123],
    [ 4.34421711e-184],
    [ 3.68913910e-168],
    [-1.88715756e-092],
    [ 2.12305495e-153],
    [ 6.32814106e-123],
    [ 1.88715756e-092],
    [ 0.00000000e+000],
    [ 0.00000000e+000],
    [ 0.00000000e+000],
    [ 0.00000000e+000],
    [ 0.00000000e+000],
    [ 0.00000000e+000]])

def main():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters =  aruco.DetectorParameters_create()
    # CORNER_REFINE_NONE, no refinement. CORNER_REFINE_SUBPIX, do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    
    drone = tellopy.Tello()

    try:
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

        # skip first 300 frames
        frame_skip = 300
        # create directory for saving image
        img_save_dir = "./output_data/imgs"
        pathlib.Path(img_save_dir).mkdir(parents=True, exist_ok=True)
        # index for saving image
        img_idx = 0
        while True:
            quit_flag = None
            for frame in container.decode(video=0):
                # Skip frames if needed
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                
                # Process frame
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                
                markerCorners, markerIds, rejectedImgPoints = aruco.detectMarkers(image, dictionary, parameters=parameters)
                
                if markerIds is not None:
                    for corners, markerId in zip(markerCorners, markerIds):
                        #print("marker_id:",markerId)
                        #print("type",type(markerId))
                        marker_size = marker_map[str(markerId[0])]['size']
                        #print("marker_size:",marker_size)
                        
                        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size*scale_factor, cameraMatrix, distCoeffs)
                        
                        # 不要なaxisを除去
                        tvec = np.squeeze(tvecs)
                        rvec = np.squeeze(rvecs)
                        
                        cv2.aruco.drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, 0.15/2);
                        
                        if markerId==0:
                            rL = Rotation.from_rotvec(rvec)
                            tL = tvec
                            rL_prime = Rotation.from_rotvec(np.array([np.pi,0,0]))*rL # まずカメラ座標を180度回転させる
                            
                            R = rL_prime.as_matrix()
                            rG = Rotation.from_matrix(R.T)
                            tG = -np.matmul(R.T,tL)
                            ypr = rG.as_euler('zyx', degrees=True)
                            
                            cv2.putText(image,"Yaw:{:6.1f} Pitch:{:6.1f} Roll:{:6.1f}".format(ypr[0],ypr[1],ypr[2]), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                    
                # convert image to RGB
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                
                cv2.imshow('Image', image)
                #cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                
                # Wait for key press (0xFF is for 64-bit support)
                key = (cv2.waitKey(1) & 0xFF)
                if key == ord('q'):
                    quit_flag = True
                    break
                elif key == ord('s'):
                    cv2.imwrite(img_save_dir+"/"+str(img_idx)+".jpg", image)
                    img_idx += 1
                
                # Calculate number of frames to skip
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
            if quit_flag == True:
                break

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
