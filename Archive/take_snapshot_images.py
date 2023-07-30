import cv2
import time


def start_snapshot_capture(seconds: str, number_of_images:str):
    #Initialise webcam
    vid = cv2.VideoCapture(0)
    seconds = int(seconds)
    number_of_images = int(number_of_images)
    #save frames
    for a in range(0,number_of_images):
        #get current frame
        ret, img = vid.read()

        time.sleep(seconds)
        #if a frame is returned
        if ret:

            #set filename to save
            filename = "./input_images/input_image_" + str(a) + '.png'

            cv2.imshow(filename, img)

            #save
            cv2.imwrite(filename, img)

            #if q is pressed, quit
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    vid.release()
    cv2.destroyAllWindows()





