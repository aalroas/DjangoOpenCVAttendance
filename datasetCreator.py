import cv2
import numpy as np
import os
faceDetect= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('http://192.168.1.20:8080/video')

id= input('Enter user id ')
Name=input('Enter student\'s name:')
sampleNum=0
while(True):
    ret, img = cap.read()
    #rgb_image = img[:, :, ::-1]
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    
    for(x,y,w,h) in faces:
        yol=os.mkdir('students/'+ id)
        cv2.imwrite(yol+str(Name+'.'+'.png')+'/')
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100);                                                                                                                                      
    cv2.imshow('face',img)
    cv2.waitKey(1)
    if(sampleNum>0):
        break
cam.release()
cv2.destroyAllWindows()
