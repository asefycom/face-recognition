import cv2
import face_recognition


shajarian01_file = 'images/shajarian01.jpg'
imgShajarian01 = face_recognition.load_image_file(shajarian01_file)
shajarian01_face_loc = face_recognition.face_locations(imgShajarian01)[0]
print(shajarian01_face_loc)
face01_rec = cv2.rectangle(imgShajarian01, (shajarian01_face_loc[3],shajarian01_face_loc[0]),
                           (shajarian01_face_loc[1], shajarian01_face_loc[2]),(255,0,255),3)


shajarian02_file = 'images/shajarian02.jpg'
imgShajarian02 = face_recognition.load_image_file(shajarian02_file)
shajarian02_face_loc = face_recognition.face_locations(imgShajarian02)[0]
print(shajarian02_face_loc)
face02_rec = cv2.rectangle(imgShajarian02, (shajarian02_face_loc[3],shajarian02_face_loc[0]),
                           (shajarian02_face_loc[1], shajarian02_face_loc[2]),(255,0,255),3)


imgShajarian01 = cv2.cvtColor(imgShajarian01,cv2.COLOR_RGB2BGR)
imgShajarian02 = cv2.cvtColor(imgShajarian02,cv2.COLOR_RGB2BGR)
cv2.imshow('shajarian training image', imgShajarian01)
cv2.imshow('shajarian test image', imgShajarian02)
cv2.waitKey(0)