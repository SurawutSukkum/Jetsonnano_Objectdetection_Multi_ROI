import cv2
from tensorflow.keras.models import load_model

import numpy as np
import os
import jetson.inference
import jetson.utils 
from datetime import datetime

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
print (DIR_PATH)
directory = f'{DIR_PATH}/OK'

# My constants
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_TYPE = "/dev/video0"


print("START")
sizeTarget = (224, 224)
np.set_printoptions(suppress=True)
dataObj = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

model_path = f'{DIR_PATH}/converted_keras/keras_model.h5'
print(model_path)
model = load_model(model_path) #path model

# Create the camera and display

camera = jetson.utils.videoSource("/dev/video0")
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while(True):
     cuda_img  = camera.Capture()
     cuda_img1 = cuda_img
     jetson.utils.cudaDeviceSynchronize()
     print(cuda_img.width)

     cv_img_rgb = jetson.utils.cudaToNumpy(cuda_img)
     cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)


     # determine the amount of border pixels (cropping around the center by half)
     crop_factor = 0.10
     crop_border = ((1.0 - crop_factor) * 0.5 * cuda_img.width,(1.0 - crop_factor) * 0.5 * cuda_img.height)

     # compute the ROI as (left, top, right, bottom)
     crop_roi = (crop_border[0], crop_border[1], cuda_img.width - crop_border[0], cuda_img.height - crop_border[1])
     #crop_roi = (int(cuda_img.width * crop_factor+460),int(cuda_img.height * crop_factor+240),int(cuda_img.width * crop_factor+420),int(cuda_img.height * crop_factor+240))
     
     # allocate the output image, with the cropped size
     imgOutput = jetson.utils.cudaAllocMapped(width=cuda_img.width * crop_factor,
                                         height=cuda_img.height * crop_factor,
                                         format=cuda_img.format)

     # crop the image to the ROI
     jetson.utils.cudaCrop(cuda_img, imgOutput, crop_roi)

     cv_img_rgb1 = jetson.utils.cudaToNumpy(imgOutput)
     cv_img_bgr1 = cv2.cvtColor(cv_img_rgb1, cv2.COLOR_RGB2BGR)
     cv2.imshow("Video Crop", cv_img_bgr1)

     # determine the amount of border pixels (cropping around the center by half)
     crop_factor1 = 0.10
     crop_border1 = ((1.0 - crop_factor1) * 0.5 * cuda_img1.width,(1.0 - crop_factor1) * 0.5 * cuda_img1.height)

     # compute the ROI as (left, top, right, bottom)
     offset = 100
     crop_roi1 = (crop_border1[0]-offset, crop_border1[1]-offset, cuda_img1.width - crop_border1[0]-offset, cuda_img1.height - crop_border1[1]-offset)

     # allocate the output image, with the cropped size
     imgOutput1 = jetson.utils.cudaAllocMapped(width=cuda_img1.width * crop_factor1,
                                         height=cuda_img1.height * crop_factor1,
                                         format=cuda_img1.format)

     # crop the image to the ROI
     jetson.utils.cudaCrop(cuda_img1, imgOutput1, crop_roi1)

     cv_img_rgb2 = jetson.utils.cudaToNumpy(imgOutput1)
     cv_img_bgr2 = cv2.cvtColor(cv_img_rgb2, cv2.COLOR_RGB2BGR)
     cv2.imshow("Video Crop1", cv_img_bgr2)

     # Change the current directory 
     # to specified directory 
     os.chdir(directory)
       
     # Filename
     dateTimeObj = datetime.now()
     timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H_%M_%S.jpg")
     filename = timestampStr
       
     # Using cv2.imwrite() method
     # Saving the image
     #cv2.imwrite(filename, cv_img_bgr1)
 
     if cv_img_bgr1 is not None:
        img_resize = cv2.resize(cv_img_bgr1,sizeTarget) #resize image      
        image_array = np.asarray(img_resize)#convert image to array
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1 #normalized image

        dataObj[0] = normalized_image_array #get frist dimention 
        prediction =  list(model.predict(dataObj)[0])#change np.ndarray to list 
        idx = prediction.index(max(prediction)) #get index is maximun value
        textColor=(0, 0, 0)
        textColor1=(255, 0, 0)
        if  prediction[idx]*100 > 98:
            if idx == 0:
                cv2.putText(cv_img_bgr, "PASS: "+str(round(prediction[idx]*100,2)) +"%", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 4,(100, 255, 100),15,cv2.FILLED)
                textColor=(100, 255, 100) 
            elif idx == 2:
                cv2.putText(cv_img_bgr, "No Object: "+str(round(prediction[idx]*100,2))+"%", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 4,(100, 100, 255),15,cv2.FILLED)
                textColor=(0, 0, 0)
        if  prediction[idx]*100 > 50:
            if idx == 1:
                cv2.putText(cv_img_bgr, "FAIL: "+str(round(prediction[idx]*100,2))+"%", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 4,(100, 100, 255),15,cv2.FILLED)
                textColor=(100, 100, 255) 
            elif idx == 2:
                cv2.putText(cv_img_bgr, "No Object: "+str(round(prediction[idx]*100,2))+"%", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 4,(100, 100, 255),15,cv2.FILLED)
                textColor=(0, 0, 0)
     cv2.rectangle(cv_img_bgr,(int(crop_roi[0]), int(crop_roi[1])), (int(crop_roi[2]), int( crop_roi[3])),textColor,5)
     cv2.rectangle(cv_img_bgr,(int(crop_roi1[0]), int(crop_roi1[1])), (int(crop_roi1[2]), int( crop_roi1[3])),textColor,5)
     cv2.imshow("Video Feed", cv_img_bgr)
     c = cv2.waitKey(1)
     if c == 27:
       break

camera.release()
cv2.destroyAllWindows()
