from ultralytics import YOLO


cosSaturationModelForTraining = YOLO('yolov8n.yaml')
#Loads the untrained model 
saturationModelForTraining = YOLO('yolov8n.yaml')
#Loads the untrained model

if __name__ == '__main__':
    cosSaturationModel = cosSaturationModelForTraining.train(data='data.yaml', epochs=100, imgsz=640, verbose=True, cos_lr=True,  hsv_s=.8725, workers=0)
    #Increases the saturation, which allows for increasing the intensity of color. May help for better alienation of tumours from the rest of the image.
    #Changes the learning rate to that of a cosine curve so that the learning rate optimizes smoother
    saturationModel = saturationModelForTraining.train(data='data.yaml', epochs=100, imgsz=640, verbose=True,  hsv_s=.8725)
    #Increases the saturation, which allows for increasing the intensity of color. May help for better alienation of tumours from the rest of the image

