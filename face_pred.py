import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recog = cv2.createLBPHFaceRecognizer()

vid = cv2.VideoCapture(0)
people={}
l = os.listdir('Faces/')
for i in l:
    l1=i.split('.')
    people[int(l1[1])]=l1[0]
recog.load('recog/recognized.yml')
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,1,0,2)
while True:
    _, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        id1,conf=recog.predict(roi_gray)
        cv2.cv.PutText(cv2.cv.fromarray(frame),people[id1],(x,y),font,(0,255,0))
    cv2.imshow('original',frame)
    if cv2.waitKey(1)==ord('a'):
        break
    
vid.release()
cv2.destroyAllWindows()
