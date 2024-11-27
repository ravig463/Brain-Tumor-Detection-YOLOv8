Overview/Breakdown of YOLOv8 Model folders:

##Data Set Folders

- test: This folder contains all the images within the test set that are being used for testing the efficacy of the model.
- train: This folder contains all the images within the training set that are being used for training the model.
- valid: This folder contains all the images within the validation set that are being used for post epoch validations and tuning.


##Models

YoloV8 Trained Models: This folder contains all the models that have been trained. They can be loaded for testing. 

##Training

YoloV8 Training Sequences: This folder contains the python scripts used in training the models (hyperparameters). This folder also contains the results from training (i.e. the charting/metrics of the validations between each epoch). Additionally, it has the YAML file used for loading model.

##Testing

YoloV8 Testing: This folder contains the testing script used to test the model against the test set. This folder also contains the runs folder which has the performance of the model during the testing validation. Additionally, it includes the YAML for testing.

__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

Overview/Breakdown of Pretrained Model folders:

The dataset has three classes of bounding boxes that each signify a different type of tumor: 0 - gliomas, 1 - meningiomas, 2 - metastatic tumors. Here is the link: https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection

This code corpus consists of the pretrained VGG-16 and MobileNetV3Large models trained using various configurations. They are explained below:

- VGG16_Initial Run | MobileNetV3Large_InitialRun: The initial run of the VGG16 and MobileNetV3Large models without any Oversampling, Early Stopping, or Dropout layers. These model have a special head to conduct bounding box regression as the output. They are configured with an Adam optimizer and a Mean Squared Error loss function. The hyperparameters used are shuffle=True, batch_size=32, epochs=100, and verbose=1.

- VGG16_Oversampling | MobileNetV3Large_Oversampling: These versions of the VGG16 and MobileNetV3Large models perform Oversampling on all of the classes.This is done by constructing a train dataset with an equal number of bounding boxes that correspond to different classes (number of class 0 bounding boxes = number of class 1 bounding boxes…). The rest of the configurations are the same as the initial runs.

- VGG16_EarlyStopping | MobileNetV3Large_EarlyStopping: These version of the VGG16 and MobileNetV3Large models perform Early Stopping by adding an EarlyStopping measure from the callbacks library as a hyperparameter. Also, only the meningiomas are oversampled to equal the number of gliomas. The rest of the configurations are the same as the initial runs.

- VGG16_SGD | MobileNetV3Large_SGD: These versions of the VGG16 and MobileNetV3Large models use the SGD optimizer instead of the Adam optimizer. Also, only the meningiomas are oversampled to equal the number of gliomas. Early stopping is not used for these trials. The rest of the configurations are the same as the initial runs.

- VGG16_Dropout | MobileNetV3Large_Dropout: These versions of the VGG16 and MobileNetV3Large models inculcate Dropout layers within the model. Also, only the meningiomas are oversampled to equal the number of gliomas. Early stopping is not used for these trials. These models use an Adam optimizer. The rest of the configurations are the same as the initial runs.

##Folder Format

Each folder corresponds to one of the ten models described above (each model architecture has five different versions):

- .csv file: This csv file contains the results of various metrics such as accuracy, loss, precision, and recall after each training epoch has been completed.

- Graphs folder: This folder contains photographs of various graphs that display the model's performance using certain metrics. The metrics tracked using these graphs include accuracy, loss, precision, and recall for training and validation (8 graphs total).

- .keras file: This file contains the stored model architecture and weights. It can be loaded to be used for testing or to make improvements on the model.

- .out file: This file tracks the progress of the model across each epoch. After each epoch is complete, it outputs the final metrics (these get stored in the csv file). In addition, the .out file also contains printed accuracies that show the model performance on the validation and test datasets.

- .py file: The Python file used to train and evaluate the model using the stated configurations and hyperparameters.

##Additional File

- GraphCreation.py: Used to generate graphs that track the various training and validation metrics of each model.

##Usage

1. Download the database with the following file structure:

- train
	- images: Has the training images
	- labels: Has the training label files (the label files have bounding box coordinates and the box’s class)

- valid
	- images: Has the validation images
	- labels: Has the validation label files (contain bounding box coordinates and the box’s class)

- test
	- images: Has the test images
	- labels: Has the test label files (contain bounding box coordinates and the box’s class)


2. Install any required dependencies or libraries that are required for the model to run. These are indicated at the top of the Python files.

3. Make sure that python/3.9.2 and python-libs/3.0 are loaded. This can be done using the following statements:

module load python/3.9.2
module load python-libs/3.0

4. Change the directory paths in the .py files to point to valid directories of training images/bounding boxes, validation images/bounding boxes, and test images/bounding boxes in the original dataset.

5. After this, the model can be run on a GPU.
