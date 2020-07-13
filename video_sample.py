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

scale_factor = 0.112/2/500

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

frame_width = 960
frame_height = 720

def main():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters =  aruco.DetectorParameters_create()
    # CORNER_REFINE_NONE, no refinement. CORNER_REFINE_SUBPIX, do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    
    
    out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
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
        
        t_ctrl_start = time.time()
        i_proc = -1
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
                
                original_image = image.copy()
                
                markerCorners, markerIds, rejectedImgPoints = aruco.detectMarkers(image, dictionary, parameters=parameters)
                
                r_cm = None
                t_cm = None
                if markerIds is not None:
                    for corners, markerId in zip(markerCorners, markerIds):
                        marker_size = marker_map[str(markerId[0])]['size']
                        
                        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size*scale_factor, cameraMatrix, distCoeffs)
                        
                        # 不要なaxisを除去
                        tvec = np.squeeze(tvecs)
                        rvec = np.squeeze(rvecs)
                        
                        cv2.aruco.drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, 0.15/2);
                        
                        if markerId[0]==0:
                            r_cm = Rotation.from_rotvec(rvec)
                            t_cm = tvec
                
                # marker +x=right, +y=up on marker
                # camera +x=right, +y=down on image
                # local: +x=right, +y=up on image
                #
                #                        rvec,tvec
                #  local  -----> camera -----> marker
                #         <-----        <-----
                #          T_cl          T_mc
                # 
                #   T_ml = T_mc*T_cl = [R^T  -R^T*t ] * [diag(1,-1,-1)  0]
                #                      [ 0    1     ]   [ 0             1]
                
                if r_cm is not None:
                    R=r_cm.as_matrix() # rotation from camera to marker
                    R_ml = np.matmul(R.T,np.array([[1,0,0],[0,-1,0],[0,0,-1]])) # rotation from marker to local
                    r_ml = Rotation.from_matrix(R_ml) 
                    t_ml = -np.matmul(R.T,t_cm) # translation from marker to local
                    
                    ypr = r_ml.as_euler('zyx', degrees=True)
                    
                    cv2.putText(image,"yaw:{:6.1f} deg   pitch:{:6.1f} deg   roll:{:6.1f} deg".format(ypr[0],ypr[1],ypr[2]), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(image,"  x:{:6.2f} m         y:{:6.2f} m        z:{:6.2f} m".format(t_ml[0],t_ml[1],t_ml[2]), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                    
                # convert image to RGB
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                
                cv2.imshow('Image', image)
                #cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                
                out.write(image)
                
                # Wait for key press (0xFF is for 64-bit support)
                key = (cv2.waitKey(1) & 0xFF)
                if key == ord('q'):
                    quit_flag = True
                    break
                elif key == ord('s'):
                    cv2.imwrite(img_save_dir+"/"+str(img_idx)+".jpg", original_image)
                    img_idx += 1
                
                # Calculate number of frames to skip
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
                
                # follow pre-defined procedures
                t_elapsed = time.time() - t_ctrl_start
                if i_proc==0 and t_elapsed<5:
                    print("waiting for takeoff")
                if i_proc==0 and t_elapsed>5:
                    drone.takeoff()
                    i_proc+=1
                elif i_proc==1 and t_elapsed>10:
                    drone.down(50)
                    i_proc+=1
                elif i_proc==2 and t_elapsed>15:
                    drone.land()
                    i_proc+=1
                elif i_proc==2 and t_elapsed>20:
                    drone.quit()
                    i_proc+=1
            
            if quit_flag == True:
                break

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        out.release()
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
