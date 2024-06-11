import cv2
from multiprocessing import Process, Value


def box_on_frame(frame):
    """
    Puts a box in the frame indicating where the model will look

    args:
        frame: the cv2 frame to 
    """
    frame = cv2.rectangle(frame, (32, 32), (64, 64), (255, 0, 0), 1)
    return frame

def main_loop():
    """
    main loop for the application
    """
    vid = cv2.VideoCapture(0) # get a file descriptor to the webcam

    while True:
        ret, frame = vid.read() # read a frame
        if not ret:
            continue
        frame = box_on_frame(frame)
        cv2.imshow("frame", frame) # show the frame
        
        # if the user presses q break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

main_loop()