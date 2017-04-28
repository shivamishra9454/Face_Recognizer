import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id1 = int(raw_input("Enter Id   = "))
name = raw_input("Enter Name = ")
count = 0
vid = cv2.VideoCapture(0)
while True:
    _, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        cv2.imwrite('C:/Users/Yashwant/Desktop/Face Recognizer/Faces/'+name+'.'+str(id1)+'.'+str(count)+'.jpg', roi_gray)
    cv2.imshow('original',frame)
    cv2.waitKey(100)
    if count>=20:
        break
vid.release()
cv2.destroyAllWindows()
