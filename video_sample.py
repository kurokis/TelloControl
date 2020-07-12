import sys
import traceback
import tellopy
import av
import cv2
import numpy
import time
import pathlib

 


def main():
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
                cv2.imshow('Original', image)
                #cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                
                # Wait for key press (0xFF is for 64-bit support)
                key = (cv2.waitKey(1) & 0xFF)
                if key == ord('q'):
                    quit_flag = True
                    break
                elif key == ord('s'):
                    cv2.imwrite(img_save_dir+"/"+str(img_idx)+".jpg", image)
                
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
