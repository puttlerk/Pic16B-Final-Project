# Pic16B-Final-Project

# Problem/Interest
	For this project we, Karl Puttler and Artem Kiryukhin,  plan to use a dataset of 87,000 images of various American Sign Language hand signs to train a model which could accurately translate the hand signs to their appropriate meanings. We believe that this project could be a stepping stone to create a more inclusive environment for deaf people, a demographic that makes up over half a million people in the US. In addition to potentially making an impact for the deaf community we want to use the knowledge that we obtained from our math degrees to implement linear algebra and various mathematical imaging techniques (e.g. convolution) in a practical manner. 
# Resources
In order to complete this project we will utilize the aforementioned dataset (https://www.kaggle.com/datasets/grassknoted/asl-alphabet) as well as Python libraries like PyTorch, NumPy, Matplotlib, among others. Unfortunately we do not have access to a powerful GPU so for our training we will make use of Google Collab’s services, specifically the virtual computing tools they provide. 

# Summary of Others Work
	Since we are planning on using a dataset from a Kaggle competition, we have collected some of the top solutions submitted during said competition. Chiefly, we focused on the following:
The first place solution, submitted by HOYSO48, achieved a public LB score (score calculated on a subset of the testing set) of 0.8 and a CV score of 0.8. In this project, the author trained a 4x seed 1D Convolutional Neural Network using TensorFlow and Google Colab TPU and then applied a Transformer, using the neural network as a “trainable tokenizer.” Beyond this, HOYSO48 outlines their efforts to avoid overfitting and in particular co-adaptation. In particular, they used Dropout with a probability of 0.8, Drop Path with a probability of 0.2, and Adversarial Weight Perturbation with a lambda of 0.2. Also useful, HOYSO48 gave a list of methods that were less effective than their final solution, in particular, they determined that Graph Convolutional Networks and that solely using Transformers both exhibited decreased performance when compared to the 1D CNN that they eventually opted for.
The fourth place solution, submitted by OHKAWA3, achieved a public LB score just under the first place score of 0.8. Once again, this author trained 1D CNNs. Of note in this solution is the author’s explanation of their preprocessing and data cleaning approaches. In particular, they outline their usage of feature points on the hands and lips when passing images and videos to their model and their use of CleanLab to remove noisy data. 
The sixth place solution, submitted by THEO VIEL, which achieved a public LB just under 0.8 and a CV just under 0.76. This solution exclusively used Transformers, in contrast to the previous two. Their summary points out a few key points. Of particular note is their assertion that the Deberta transformer model was more effective than the Bert model, although it required them to rewrite the attention layer. Another is that they disagree with the augmentations given in the other two models. Namely, they found Dropout and Drop Frame augmentations to provide little improvement in model performance.

# Tools/Skills
On a more technical level we are planning to utilize Convolutional Neural Networks (CNN) to train our model. CNN is the most common type of neural network used for image classification tasks since it allows the model to reduce the amount of weights necessary to learn as well as cleverly deciphering patterns in the images with few simple mathematical operations. Picking the appropriate architecture for this type of network could pose a problem so we plan to draw inspiration from previous research literature about this subject. In addition this project will require processing a relatively large dataset so we will need to familiarize ourselves with techniques for handling a dataset of such scale. Clearly basic Pandas will not be sufficient for this project and we instead will need to make use of PyTorch’s Datasets and DataLoader functionality to efficiently work with the data at hand. Additionally if the project timeline is more forgiving than we anticipated we could use the library OpenCV in order to process live images and translate hand gestures in real time. 

# What we will learn
From this project we hope to learn many things relevant to building and implementing a successful machine learning model. Firstly we hope to familiarize ourselves with working on larger datasets and preprocessing data. This knowledge is surely transferable to any machine learning project and will help us out tremendously in our future endeavors. Additionally we will need to gain a deeper understanding of CNNs and Neural Networks in general to make sure that our project will run as smoothly as possible. This project will prove to be of great value in both of our understanding of machine learning if we succeed at learning the two crucial skills mentioned above. Although the concentration of this topic is on training models we believe that we will learn broader skills related to programming such as utilizing git and GitHub to maneuver version control as well as general project managing protocols i.e. tasks allocation and effective communications. 

# Roles and Timeline
Karl:
Week of 05/05 to 05/11: Plan out method for image/video preprocessing and model type
Week of 05/12 to 05/18: Implement image/video pre-processing
Week of 05/19 to 05/25: Train Model and start documenting our methods for performing prec-processing and model preparation 
Week of 05/26 to 06/01: Continue writing explanations as well as figuring out how to interpret our model’s performance
Week of 06/02 to 06/08: Prepare for the presentation/Present
Artem:
Week of 05/05 to 05/11: Plan out image/video preprocessing and model type
Week of 05/12 to 05/18: Figuring out model architecture 
Week of 05/19 to 05/25: Train models and start the process of hyperparameter optimization as well as refining the overall architecture of the network 
Week of 05/26 to 06/01: Create visuals to more effectively interpret the results of the model and where there is room for improvement
Week of 06/02 to 06/08: Prepare for the presentation/Present
