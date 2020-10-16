from cv2 import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'C:\\Users\\Rakesh\\Desktop\\pyproject\\face_samples\\'
files = [f for f in listdir(data_path) if isfile(join(data_path,f))]
train_data, labels = [], []

for i,f in enumerate(files):
    img_path = data_path + files[i]
    images = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    train_data.append(np.asarray(images,dtype=np.uint8))
    labels.append(i)

labels = [i for i in range(len(train_data))]

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(train_data),np.asarray(labels))

print('model training complete')

face_classifier = cv2.CascadeClassifier('C:\\Users\\Rakesh\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray,1.3,5)

    if face is():
        return img,[]

    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,234,150),2)
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi,(200,200))

    return img,roi

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 *(1-(result[1])/300))
            display = str(confidence)+ '%confidence it is user'
        cv2.putText(img,display,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,45),2)

        if confidence > 75:
            cv2.putText(img,'unlock',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(50,255,45),2)
            cv2.imshow('face',img)

        else:
            cv2.putText(img,'locked',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow('face',img)

    except:
        cv2.putText(img,'face not found',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('face',img)
        pass
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

