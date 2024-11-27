Overview/Breakdown of folders:

Data Set Folders:

test--this folder contains all the images within the test set that are being used for testing the efficacy of the model
train--this folder contains all the images within the training set that are being used for training the model
valid--this folder contains all the images within the validation set that are being used for post epoch validations and tuning


Models:
YoloV8 Trained Models: this folder contains all the models that have been trained. They can be loaded for testing. 

Training:
YoloV8 Training Sequences: this folder contains the python scripts used in training the models (hyperparameters). This folder also contains the results from training (i.e. the charting/metrics of the validations between each epoch). Also has the YAML file used for loading model.

Testing:
YoloV8 Testing: this folder contains the testing script used to test the model against the test set. This folder also contains the runs folder, which has the performance of the model during the testing validation. Also has includes the YAML for testing.