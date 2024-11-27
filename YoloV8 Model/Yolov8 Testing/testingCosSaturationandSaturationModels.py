from ultralytics import YOLO
cosSaturationModelForTraining = YOLO('cosSaturationModel.pt')
#Loads the trained model 
saturationModelForTraining = YOLO('saturationModel.pt')
#Loads the trained model
if __name__ == '__main__':
    cosSaturationModel = cosSaturationModelForTraining.val(data='testing.yaml')
    #Increases the saturation, which allows for increasing the intensity of color. May help for better alienation of tumours from the rest of the image.
    #Changes the learning rate to that of a cosine curve so that the learning rate optimizes smoother
    saturationModel = saturationModelForTraining.val(data='testing.yaml')
    #Increases the saturation, which allows for increasing the intensity of color. May help for better alienation of tumours from the rest of the image






