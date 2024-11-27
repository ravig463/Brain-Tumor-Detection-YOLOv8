from ultralytics import YOLO
ModelOne = YOLO('modelOne.pt')
#Load the Model 
ModelTwo = YOLO('modelTwo.pt')
#Load the Model 
ModelThree = YOLO('modelThree.pt')
#Load the Model 
ModelFour = YOLO('modelFour.pt')
#Load the Model 
if __name__ == '__main__':
    ModelOneTrained = ModelOne.val(data='testing.yaml')
    #Utilized the stochastic gradient optimizer. Has a higher focal loss to address imbalance. Image size preprocessing is 140x140. Saturation is higher.
    ModelTwoTrained = ModelTwo.val(data='testing.yaml')
    #Model uses stochastic gradient optimizer. A focal loss of 3, which is 1.5 above default. Image resolution of 140x140. A hue augmentation of .1. Working threads is 16 for training.
    #The classification loss is set to .8 making correct classifications more important. Saturation augmentation set to .85.
    ModelThreeTrained = ModelThree.val(data='testing.yaml')
    #Utilized the stochastic gradient optimizer. Has a higher focal loss to address imbalance. Image size preprocessing is 640x640. Saturation is higher.
    ModelFourTrained = ModelFour.val(data='testing.yaml')
    #Utilized the Adam optimizer. Has a higher focal loss to address imbalance. Image size preprocessing is 140x140. Saturation is higher. The learning rate starts lower for the Adam Optimizer
    

