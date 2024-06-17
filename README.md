# Pic16B-Final-Project Proposal

## Problem/Interest
For this project we, Karl Puttler and Artem Kiryukhin,  plan to use a dataset of 87,000 images of various American Sign Language hand signs to train a model which could accurately translate the hand signs to their appropriate meanings. We believe that this project could be a stepping stone to create a more inclusive environment for deaf people, a demographic that makes up over half a million people in the US. In addition to potentially making an impact for the deaf community we want to use the knowledge that we obtained from our math degrees to implement linear algebra and various mathematical imaging techniques (e.g. convolution) in a practical manner. 
## Resources
In order to complete this project we will utilize the aforementioned dataset (https://www.kaggle.com/datasets/grassknoted/asl-alphabet) as well as Python libraries like PyTorch, NumPy, Matplotlib, among others. Unfortunately we do not have access to a powerful GPU so for our training we will make use of Google Collab’s services, specifically the virtual computing tools they provide. 

## Summary of Others Work
Since we are planning on using a dataset from a Kaggle competition, we have collected some of the top solutions submitted during said competition. Chiefly, we focused on the following:

### First Place Solution
The first place solution, submitted by HOYSO48, achieved a public LB score (score calculated on a subset of the testing set) of 0.8 and a CV score of 0.8. In this project, the author trained a 4x seed 1D Convolutional Neural Network using TensorFlow and Google Colab TPU and then applied a Transformer, using the neural network as a “trainable tokenizer.” Beyond this, HOYSO48 outlines their efforts to avoid overfitting and in particular co-adaptation. In particular, they used Dropout with a probability of 0.8, Drop Path with a probability of 0.2, and Adversarial Weight Perturbation with a lambda of 0.2. Also useful, HOYSO48 gave a list of methods that were less effective than their final solution, in particular, they determined that Graph Convolutional Networks and that solely using Transformers both exhibited decreased performance when compared to the 1D CNN that they eventually opted for.

Link to first place solution: https://www.kaggle.com/competitions/asl-signs/discussion/406684

### Fourth Place Solution
The fourth place solution, submitted by OHKAWA3, achieved a public LB score just under the first place score of 0.8. Once again, this author trained 1D CNNs. Of note in this solution is the author’s explanation of their preprocessing and data cleaning approaches. In particular, they outline their usage of feature points on the hands and lips when passing images and videos to their model and their use of CleanLab to remove noisy data. 


Link to second place solution: https://www.kaggle.com/competitions/asl-signs/discussion/406673

### Sixth Place Solution
The sixth place solution, submitted by THEO VIEL, which achieved a public LB just under 0.8 and a CV just under 0.76. This solution exclusively used Transformers, in contrast to the previous two. Their summary points out a few key points. Of particular note is their assertion that the Deberta transformer model was more effective than the Bert model, although it required them to rewrite the attention layer. Another is that they disagree with the augmentations given in the other two models. Namely, they found Dropout and Drop Frame augmentations to provide little improvement in model performance.


Link to third place solution: https://www.kaggle.com/competitions/asl-signs/discussion/406537

## Tools/Skills
On a more technical level we are planning to utilize Convolutional Neural Networks (CNN) to train our model. CNN is the most common type of neural network used for image classification tasks since it allows the model to reduce the amount of weights necessary to learn as well as cleverly deciphering patterns in the images with few simple mathematical operations. Picking the appropriate architecture for this type of network could pose a problem so we plan to draw inspiration from previous research literature about this subject. In addition this project will require processing a relatively large dataset so we will need to familiarize ourselves with techniques for handling a dataset of such scale. Clearly basic Pandas will not be sufficient for this project and we instead will need to make use of PyTorch’s Datasets and DataLoader functionality to efficiently work with the data at hand. Additionally if the project timeline is more forgiving than we anticipated we could use the library OpenCV in order to process live images and translate hand gestures in real time. 

## What we will learn
From this project we hope to learn many things relevant to building and implementing a successful machine learning model. Firstly we hope to familiarize ourselves with working on larger datasets and preprocessing data. This knowledge is surely transferable to any machine learning project and will help us out tremendously in our future endeavors. Additionally we will need to gain a deeper understanding of CNNs and Neural Networks in general to make sure that our project will run as smoothly as possible. This project will prove to be of great value in both of our understanding of machine learning if we succeed at learning the two crucial skills mentioned above. Although the concentration of this topic is on training models we believe that we will learn broader skills related to programming such as utilizing git and GitHub to maneuver version control as well as general project managing protocols i.e. tasks allocation and effective communications. 

## Roles and Timeline
### Karl:
- Week of 05/05 to 05/11: Plan out method for image/video preprocessing and model type
- Week of 05/12 to 05/18: Implement image/video pre-processing
- Week of 05/19 to 05/25: Train Model and start documenting our methods for performing prec-processing and model preparation 
- Week of 05/26 to 06/01: Continue writing explanations as well as figuring out how to interpret our model’s performance
- Week of 06/02 to 06/08: Prepare for the presentation/Present

### Artem:
- Week of 05/05 to 05/11: Plan out image/video preprocessing and model type
- Week of 05/12 to 05/18: Figuring out model architecture 
- Week of 05/19 to 05/25: Train models and start the process of hyperparameter optimization as well as refining the overall architecture of the network 
- Week of 05/26 to 06/01: Create visuals to more effectively interpret the results of the model and where there is room for improvement
- Week of 06/02 to 06/08: Prepare for the presentation/Present

# Real-Time ASL Hand Sign Classification Using Convolutional Neural Networks (Project Summary)
## Project Introduction
Our project aimed to develop a model that can accurately translate American Sign Language (ASL) hand signs into their corresponding meanings. We used a dataset of 87,000 images to train this model and implement it into a real-time classification framework using a webcam. This project holds potential for enhancing online video call accessibility for deaf individuals.
## Data Processing
We used a Kaggle dataset containing 87,000 images of ASL hand signs for the English alphabet, space, delete, and no sign (https://www.kaggle.com/datasets/grassknoted/asl-alphabet). The images had almost identical backgrounds, lighting, and camera distance, which is suboptimal for real-time input that could differ in many ways. 
To process the data, we resized the images to 32x32x3 for LeNet5 and 227x227x3 for AlexNet. We applied data augmentation techniques, including random rotations and color adjustments, to introduce variety and enhance model robustness. Additionally, we normalized the pixel values to accelerate gradient descent convergence.
## Designing Webcam Input Program
Using OpenCV we created a script to capture real-time video input. A 200x200 pixel section of each frame was extracted and fed into the classification pipeline. This required the user to sign in a specific location on the frame for accurate classification.
Model Architectures and Performance
## LeNet5 Model:
Architecture: Two convolutional layers followed by max pooling, then three fully connected layers (60,000 weights)
Performance: Achieved 85-90% accuracy but struggled with signs differing by subtle finger placements due to significant compression from 200x200 to 32x32 pixels.
Conclusion: Computational simplicity led to fast training but inadequate accuracy for nuanced sign distinctions.
## AlexNet Model:
Architecture: Five convolutional layers with max pooling and dropout layers, followed by fully connected layers (60 million weights)
Performance: Achieved nearly 97% accuracy on training and validation data. Showed significant improvement in distinguishing similar signs compared to LeNet5.
Conclusion: Though training was slower, AlexNet provided better generalization and accuracy for real-time sign classification.
## Model Limitations
Our model performs well in controlled conditions but struggles with varying background colors, requiring users to sign within a predefined box. To address this, future work could include:
Image Matting: To separate hand signs from backgrounds.
Advanced Models: Using Mask R-CNN for object segmentation.
Transfer Learning: From models trained on diverse datasets.
Hand Position Detection: Using datasets like EgoHands and HGR1 to identify hand locations dynamically.
## Conclusion
Our project successfully developed a real-time ASL hand sign classification system with promising accuracy using AlexNet. While effective, the model's dependency on uniform backgrounds and fixed hand positions proves to be quite limiting. You can view the test run of our program here: https://youtu.be/AkgTBekx00Y. 
