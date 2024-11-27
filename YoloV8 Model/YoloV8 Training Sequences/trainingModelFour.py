from ultralytics import YOLO
ModelFour = YOLO('yolov8n.yaml')
#Load the Model 
if __name__ == '__main__':
    ModelFourTrained = ModelFour.train(data='data.yaml', epochs=100, save=True,optimizer='Adam',lr0=.001, dfl=1.8, imgsz=140, verbose=True,  hsv_s=.8725, plots=True)
        #Utilized the Adam optimizer. Has a higher focal loss to address imbalance. Image size preprocessing is 140x140. Saturation is higher. The learning rate starts lower for the Adam Optimizer
        