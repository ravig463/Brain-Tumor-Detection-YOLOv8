from ultralytics import YOLO
ModelOne = YOLO('yolov8n.yaml')
#Load the Model 
ModelTwo = YOLO('yolov8n.yaml')
#Load the Model 
ModelThree = YOLO('yolov8n.yaml')
#Load the Model 
if __name__ == '__main__':
    ModelOneTrained = ModelOne.train(data='data.yaml', epochs=300, save=True,optimizer='SGD', dfl=1.7, imgsz=140, verbose=True,  hsv_s=.8725, plots=True)
    #Utilized the stochastic gradient optimizer. Has a higher focal loss to address imbalance. Image size preprocessing is 140x140. Saturation is higher.
    ModelTwoTrained = ModelTwo.train(data='data.yaml', epochs=300, save=True,optimizer='SGD', dfl=3, imgsz=140, hsv_h=.1,verbose=True, workers=16, cls=.8, hsv_s=.85, plots=True)
    #Model uses stochastic gradient optimizer. A focal loss of 3, which is 1.5 above default. Image resolution of 140x140. A hue augmentation of .1. Working threads is 16 for training.
    #The classification loss is set to .8 making correct classifications more important. Saturation augmentation set to .85.
    ModelThreeTrained = ModelThree.train(data='data.yaml', epochs=100,optimizer='SGD', imgsz=640, save=True, verbose=True,  dfl=1.9, hsv_s=.8725, plots=True)
    #Utilized the stochastic gradient optimizer. Has a higher focal loss to address imbalance. Image size preprocessing is 640x640. Saturation is higher.
    

