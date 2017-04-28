import os
import cv2
import numpy as np
from PIL import Image

recog = cv2.createLBPHFaceRecognizer()
path = 'Faces/'

def getimgpaths(path):
    paths = os.listdir(path)
    faces = []
    id1 = []
    for i in paths:
        img = Image.open(path+i).convert('L')
        img_to_np = np.array(img, 'uint8')
        faces.append(img_to_np)
        ids = i.split('.')[1]
        id1.append(int(ids))
        cv2.imshow('Wait, I am Training...',img_to_np)
        cv2.waitKey(10)
    return np.array(id1), faces

id1,faces = getimgpaths(path)
recog.train(faces,id1)
recog.save('recog/recognized.yml')
print("Yay! Training is Complete!")
cv2.destroyAllWindows()
