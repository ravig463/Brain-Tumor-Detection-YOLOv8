#!/usr/bin/env python
# coding: utf-8

# In[88]:


'''
Author: Ravi Gadgil

Python file to generate graphs of the following training and validation metrics for all models: accuracy, loss,
precision, and recall.
'''

#Import all necessary modules and libraries to complete the task
import matplotlib.pyplot as plt
import pandas as pd

#Path to the csv file which contains all of the training and validation metrics
csv_filepath = "/data/groups/gomesr/REU_2023/RaviGadgil/Untitled Folder/MobileNetV3Large_Dropout/MobileNetV3Large_Dropout.csv"

#Stores all of the data from the csv file with all of the metrics
csv_data = pd.read_csv(csv_filepath)

#Convert a specific metric tracked across multiple epochs represented as a column in the csv file to a list.
metric = csv_data['val_recall'].tolist()

#Store a sequence of numbers which represent the number of epochs completed during training
epochs = range(0, 100)

#Plot the graph with the metric on the y-axis and epochs on the x-axis
plt.plot(epochs, metric)

plt.xlabel('Epochs')
plt.ylabel('Recall')

plt.title("Validation Recall")

plt.show()


# In[ ]:





# In[ ]:




