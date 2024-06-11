import cv2
from multiprocessing import Process, Value


def quit_loop(quit_flag):
    """
    prompts the user to quit the program

    args:
        quit_flag: ctype int indicating that looping should be terminated 
    """

    assert(quit_flag.value == 0) # quit_flag should be 0 when we start

    while (not quit_flag): # ask the user if they want to quit repeatedly until they do
        # get a user input as a string
        user_input = str(input("Quit y(es)/n(o): "))
        # if the string is "y" or "yes" set quit_flag to 1
        if (user_input == "y" or user_input == "yes"):
            quit_flag.value = 1

def capture_loop(quit_flag):
    """
    gets and displays video from the webcam

    args:
        quit_flag: ctype int indicating that looping should be terminated 
    """

    assert(quit_flag.value == 0) # quit_flag should be 0 when we start

    cap = cv2.VideoCapture(0) # get file descriptor for webcam

    while (not quit_flag and cap.isOpened()): # while the user hasn't quite and the webcam is open

        ret, frame = cap.read() # read a frame
        if (ret): # if the read was successful
            cv2.imshow("frame", frame) # show the frame
            print(type(frame))
        
    cap.release() # close the webcam file descriptor
    cv2.destroyAllWindows() # close the windows we made


def box_on_frame(frame):
    """
    Puts a box in the frame indicating where the model will look

    args:
        frame: the cv2 frame to 
    """

def main_loop():
    """
    main loop for the application
    """

    if (__name__ == "__main__"):

        # initialize quit_flag
        quit_flag = Value('i', 0)

        # start the quit_loop and capture loop
        q_loop = Process(target = quit_loop, args = (quit_flag,)).start()
        c_loop = Process(target = capture_loop, args = (quit_flag,)).start()

        # join with quit_loop and capture_loop
        c_loop.join()
        q_loop.join()

main_loop()