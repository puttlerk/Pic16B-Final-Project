import cv2
import numpy as np
import torch
from torchvision import transforms
import PIL
from scipy import stats

from dataProcessing import test_transform
from models import LeNet5

WEBCAM_TRANSFORM = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

labels_names = [chr(ord('A') + i) for i in range(26)]
labels_names.extend(["del","nothing","space"])

def predict_on_frame(frame, model):
    """
    Randomly applies a random rotation and brightness/contrast shift
    to the frame 100 times. Then for each iteration, makes a prediction.
    Takes the mode of the predictions and returns that. 

    @params:
        frame, PIL image representing the image
        model, pytorch neural network with which to predict
    @returns:
        number representing the predicted sign of the image
    """
    # make a bunch of predictions on slightly modified frame
    frame= PIL.Image.fromarray(frame)
    frame = WEBCAM_TRANSFORM(frame)
    output = model(frame)
    _, pred = torch.max(output.data, 1)
    pred = pred.item()
    pred = labels_names[pred]
    return pred



def main_loop():
    """
    main loop for the application

    @params:
        model_type: string, the type of model to use
        model_path: string, path to the saved model weights
    """
    # get the trained model weights
    model = LeNet5()
    model.load_state_dict(torch.load("NewImageRecognitionModel.pth"))


    # get a file descriptor to the webcam
    vid = cv2.VideoCapture(0)
    frame_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_middle_x = frame_width / 2
    frame_middle_y = frame_height / 2

    # read frames from webcam and predict with model
    while True:
        ret, frame = vid.read() # read a frame
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame = cv2.rectangle(frame,
                              (int(frame_middle_x - frame_width / 6), int(frame_middle_y - frame_width / 6)),
                              (int(frame_middle_x + frame_width / 6), int(frame_middle_y + frame_width / 6)),
                              (255, 0, 0),
                              2)
        cv2.imshow("frame", frame) # show the frame
        
        # crop out the part that we use for prediction
        pred_frame = cv2.getRectSubPix(frame, (int(frame_middle_x), int(frame_middle_y)), (int(frame_width / 3), int(frame_height / 3)))
        cv2.imshow("pred_frame", pred_frame)
        # apply the model to the frame
        pred = predict_on_frame(pred_frame, model)
        print("\rPrediction: ", pred)
        
        # if the user presses q break the loop
        while True:
            if cv2.waitKey(300) & 0xFF == ord('w'):
                break

        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

    # close file descriptor and windows
    vid.release()
    cv2.destroyAllWindows()

main_loop()