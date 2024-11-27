#!/usr/bin/env python
# coding: utf-8

'''
Author: Ravi Gadgil

Python file to train and evaluate a MobileNetV3Large pretrained model on the Brain Tumor Dataset found on Kaggle.

The tumor classes are as follows: 0 - gliomas, 1 - meningiomas, 2 - metastatic tumors.
'''

# In[36]:

#Import all necessary modules and libraries needed to complete the task

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Large
from keras.metrics import Accuracy, Precision, Recall, AUC, F1Score
import pandas as pd


# In[37]:

'''
Method to load the images and bounding boxes from the training, validation, and test datasets. These will later be used to train
and evaluate the VGG16 model.
'''

def load_data(data_dir):
    
    #Create lists to store the validation or test images and their bounding box coordinates.
    
    images = []
    boxes = []
    
    #For-loop to iterate through all of the image and label files. The label files contain the bounding box coordinates along 
    #with the corresponding class (0, 1, or 2).

    for file in os.listdir(os.path.join(data_dir, "images")):
        
        #Set the image file path and label file path.
        
        if file.endswith('.jpg'):
            image_path = os.path.join(data_dir, "images", file)
            label_path = os.path.join(data_dir, "labels", file.replace('.jpg', '.txt'))
            
            #Process the label files and extract the important components to create class-labeled bounding boxes for each image.
            
            if os.path.isfile(label_path):
                with open(label_path, 'r') as label_file:
                    line = label_file.readline().strip()

                    if not line:
                        continue
                    
                    #Extract the class label and coordinates of the bounding box for a specific image.
                    
                    elements = line.split(' ')
                    
                    x_one = float(elements[1])
                    y_one = float(elements[2])
                    x_two = float(elements[3])
                    y_two = float(elements[4])
                    
                    #Create the bounding box for a particular image using the extracted coordinates as a tuple, and store this
                    #tuple into the boxes list.
                    
                    boxes.append((x_one, y_one, x_two, y_two))

                #Load the validation or test image with the appropriate size (224 by 224 pixels), convert it into a NumPy array, 
                #and add it to the array with all of the validation or test images.
                
                images.append(img_to_array(load_img(image_path, target_size=(224, 224))))
    
    #Return the images and bounding box coordinates as NumPy arrays that are of the float32 data-type. The image NumPy array is
    #divided by 255 to perform Min-Max Normalization on the pixel values of the validation and test images. 
    
    return np.array(images,dtype="float32")/255.0, np.array(boxes, dtype="float32")

#Variables to store the path of the datasets which contain the training, validation, and test images and bounding boxes.

train_dataset_dir = '/data/groups/gomesr/REU_2023/RaviGadgil/Untitled Folder/training'
val_dataset_dir = '/data/groups/gomesr/REU_2023/RaviGadgil/Untitled Folder/val'
test_dataset_dir = '/data/groups/gomesr/REU_2023/RaviGadgil/Untitled Folder/test'

#Loads the training images in array format, training image bounding boxes, validation images, validation image bounding boxes, #test images, and test image bounding boxes into variables that can be utilized later to complete the task.

train_images, train_boxes = load_data(train_dataset_dir)
val_images, val_boxes = load_data(val_dataset_dir)
test_images, test_boxes = load_data(test_dataset_dir)

# In[38]:

#Instantiate the MobileNetV3Large model with pretrained imagenet weights, an input_shape that is equal to the resized image size, #and no 3 fully-connected layers near the top of the network.

mobilenet = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Set trainable to false to freeze the layer's variables. This is done to only utilize the pretrained processing of the imagenet 
#weights for the training and evaluation of the MobileNetV3Large model. Also, it is used to keep the training time manageable #because it requires too many computational resources (ex. time, GPUs, CPU cores) otherwise.

mobilenet.trainable = False

#Flatten the output layer to simplify the model architecture and allow for compatiability with the dense layers.

output_layer = mobilenet.output

output_layer = Flatten()(output_layer)

#Add four fully connected Dense layers to conduct bounding box regression. This allows for the accurate prediction of the tumors
#within the image as the predicted coordinates get manipulated to be closer to the true coordinates. 

bounding_box_head = Dense(128, activation="relu")(output_layer)
bounding_box_head = Dense(64, activation="relu")(bounding_box_head)
bounding_box_head = Dense(32, activation="relu")(bounding_box_head)
bounding_box_head = Dense(4, activation="sigmoid")(bounding_box_head)

#Create a model that takes in the VGG model and outputs predictions for the bounding box coordinates of each image.

model = Model(inputs=mobilenet.input, outputs=bounding_box_head)


# In[39]:

#Instantiate an Adam optimizer.

opt = Adam()

#Configure the model with the Mean Squared Error loss function, Adam optimizer, and various metrics.

model.compile(optimizer=opt, loss="mean_squared_error", metrics=['accuracy', 'precision', 'recall', 'auc', 'f1_score'])

#Train the model and store the results (the various metrics) of the training and validation for each epoch in the history  #variable to be extracted later. Verbose set to 1 to show the progress bar of the model being trained per epoch. Shuffle set to
#True to shuffle the training data before each epoch.

history = model.fit(
    train_images, train_boxes,
    validation_data=(val_images, val_boxes),
    shuffle=True,
    batch_size=32,
    epochs=100,
    verbose=1
)

#Write the metrics of the model training to a csv file to be used later.

df = pd.DataFrame(history.history)
df.to_csv('MobileNetV3Large_ObjDet.csv') 

#Evaluate the model on the validation dataset. Print out the validation accuracy.

val_accuracy = model.evaluate(val_images, val_boxes)[1]
print(f'Validation Accuracy: {val_accuracy}')

#Evaluate the model on the test dataset. Print out the test accuracy.

test_accuracy = model.evaluate(test_images, test_boxes)[1]
print(f'Test Accuracy: {test_accuracy}')

#Save the entire model and its weights into a file for later use.

model.save('MobileNetV3Large_ObjDet.keras')


# In[ ]:




