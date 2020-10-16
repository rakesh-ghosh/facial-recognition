from cv2 import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('C:\\Users\\Rakesh\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

def face_extrator(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img_gray,1.3,6)

    if faces is():
        return None
    
    for(x,y,w,h) in faces:
        crop_face = img[y:y+h, x:x+w]

    return crop_face


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extrator(frame) is not None:
        count+=1
        face = cv2.resize(face_extrator(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_name_path = 'face_samples\\user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,45),2)

        cv2.imshow('face cropped',face)
    else:
        print('face not found')
        pass

    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print('collecting samples complete')

