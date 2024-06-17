# Real-Time ASL Hand Sign Classification Using Convolutional Neural Networks
### Project Introduction
Our project aimed to develop a model that can accurately translate American Sign Language (ASL) hand signs into their corresponding meanings. We used a dataset of 87,000 images to train this model and implement it into a real-time classification framework using a webcam. This project holds potential for enhancing online video call accessibility for deaf individuals.
### Data Processing
We used a Kaggle dataset containing 87,000 images of ASL hand signs for the English alphabet, space, delete, and no sign (https://www.kaggle.com/datasets/grassknoted/asl-alphabet). The images had almost identical backgrounds, lighting, and camera distance, which is suboptimal for real-time input that could differ in many ways. 
To process the data, we resized the images to 32x32x3 for LeNet5 and 227x227x3 for AlexNet. We applied data augmentation techniques, including random rotations and color adjustments, to introduce variety and enhance model robustness. Additionally, we normalized the pixel values to accelerate gradient descent convergence.
### Designing Webcam Input Program
Using OpenCV we created a script to capture real-time video input. A 200x200 pixel section of each frame was extracted and fed into the classification pipeline. This required the user to sign in a specific location on the frame for accurate classification.
### Model Architectures and Performance
#### LeNet5 Model:
* Architecture: Two convolutional layers followed by max pooling, then three fully connected layers (60,000 weights)
* Performance: Achieved 85-90% accuracy but struggled with signs differing by subtle finger placements due to significant compression from 200x200 to 32x32 pixels.
* Conclusion: Computational simplicity led to fast training but inadequate accuracy for nuanced sign distinctions.
#### AlexNet Model:
* Architecture: Five convolutional layers with max pooling and dropout layers, followed by fully connected layers (60 million weights)
* Performance: Achieved nearly 97% accuracy on training and validation data. Showed significant improvement in distinguishing similar signs compared to LeNet5.
* Conclusion: Though training was slower, AlexNet provided better generalization and accuracy for real-time sign classification.
### Model Limitations
Our model performs well in controlled conditions but struggles with varying background colors, requiring users to sign within a predefined box. To address this, future work could include:
Image Matting: To separate hand signs from backgrounds.
Advanced Models: Using Mask R-CNN for object segmentation.
Transfer Learning: From models trained on diverse datasets.
Hand Position Detection: Using datasets like EgoHands and HGR1 to identify hand locations dynamically.
### Conclusion
Our project successfully developed a real-time ASL hand sign classification system with promising accuracy using AlexNet. While effective, the model's dependency on uniform backgrounds and fixed hand positions proves to be quite limiting. You can view the test run of our program here: https://youtu.be/AkgTBekx00Y. 

