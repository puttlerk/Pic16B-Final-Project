import cv2
from multiprocessing import Process, Value

import torch

from models import *


def get_model(type, path):
    model
    if type == "le":
        model = LeNet5
    if type == "alex":
        model = AlexNet
    else:
        raise(TypeError("cannot recognize model type"))
    model = model.load_state_dict(torch.load(path))
    return model


def main_loop(model_type, model_path):
    """
    main loop for the application

    @params:
        model_type: string, the type of model to use
        model_path: string, path to the saved model weights
    """
    # get the model
    model = get_model("le", "../ImageRecognitionModel.pth")
    model.eval()
    
    # get a file descriptor to the webcam
    vid = cv2.VideoCapture(0) 

    # read frames from webcam and predict with model
    while True:
        ret, frame = vid.read() # read a frame
        if not ret:
            continue
        cv2.imshow("frame", frame) # show the frame
        
        # apply the model to the frame
        pred = model(frame)
        
        # if the user presses q break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close file descriptor and windows
    vid.release()
    cv2.destroyAllWindows()

main_loop()