import cv2
import numpy as np
import torch
from torch import transforms
import PIL
from scipy import stats

from dataProcessing import test_transform
from models import LeNet5

WEBCAM_TRANSFORM = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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
    predictions = np.zeros(100)
    for i, prediction in enumerate(predictions):
        frame = WEBCAM_TRANSFORM
        output = model(frame)
        _, pred = torch.max(output.data, 1)
        predictions[i] = pred

    # take the most frequent prediction
    prediction = stats.mode(predictions)
    return prediction


def main_loop():
    """
    main loop for the application

    @params:
        model_type: string, the type of model to use
        model_path: string, path to the saved model weights
    """
    # get the trained model weights
    model = LeNet5
    model.load_state_dict(torch.load("../ImageRecognitionModel.pth"))

    # get a file descriptor to the webcam
    vid = cv2.VideoCapture(0) 

    # read frames from webcam and predict with model
    while True:
        ret, frame = vid.read() # read a frame
        if not ret:
            continue
        frame = cv2.rectangle(frame, (172, 172), (428, 428), (255, 0, 0), 2)
        cv2.imshow("frame", frame) # show the frame
        
        # crop out the part that we use for prediction
        pred_frame = cv2.getRectSubPix(frame, (256, 256), (300, 300))       
        pred_frame = PIL.Image.from_numpy(pred_frame)

        # apply the model to the frame
        pred = predict_on_frame(pred_frame, model)
        print("\rPrediction: ", pred)
        
        # if the user presses q break the loop
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # close file descriptor and windows
    vid.release()
    cv2.destroyAllWindows()

main_loop()