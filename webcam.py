import cv2
import numpy as np
import torch
from torchvision import transforms
import PIL

from workflow_functions import *


WEBCAM_TRANSFORM = transforms.Compose([
    transforms.Resize((227,227)),
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = PIL.Image.fromarray(frame)


    frame = WEBCAM_TRANSFORM(frame)
    image = transforms.ToPILImage()(frame)
    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('transformed', image)

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
    model = load_model("AlexNet30Epoch.pth")

    # get a file descriptor to the webcam
    vid = cv2.VideoCapture(0)
    frame_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_middle_x = frame_width / 2
    frame_middle_y = frame_height / 2
    
    top_left_x = int(frame_middle_x - 1200 / 6)
    top_left_y = int(frame_middle_y - 1200 / 6)
    bot_right_x = int(frame_middle_x + 1200 / 6)
    bot_right_y = int(frame_middle_y + 1200 / 6)

    number_to_display = 42  # Example number
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (50, 50)  # Position where the text will be displayed
    font_scale = 1
    font_color = (0, 255, 0)  # Green color in BGR
    line_type = 2


    # read frames from webcam and predict with model
    while True:
        ret, frame = vid.read() # read a frame
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.rectangle(frame,
                              (top_left_x, top_left_y),
                              (bot_right_x, bot_right_y),
                              (255, 0, 0),
                              2)

         # show the frame
        
        # crop out the part that we use for prediction
        pred_frame = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]
        cv2.imshow("pred_frame", pred_frame)
        # apply the model to the frame
        pred = predict_on_frame(pred_frame, model)
        print("\rPrediction: ", pred)

        cv2.putText(frame, str(pred),
                    position,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        cv2.imshow("frame", frame)

        # if the user presses q break the loop
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

    # close file descriptor and windows
    vid.release()
    cv2.destroyAllWindows()

main_loop()