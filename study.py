import cv2
import face_recognition

shajarian01_file = 'images/shajarian01.jpg'

imgShajarian01 = face_recognition.load_image_file(shajarian01_file)
# imgShajarian01CV = cv2.imread(shajarian01_file)
#
# print(imgShajarian01)
# print('-----------')
# print(imgShajarian01CV)
shajarian01_face_loc = face_recognition.face_locations(imgShajarian01)[0]
print(shajarian01_face_loc)

face_rec = cv2.rectangle(imgShajarian01, (shajarian01_face_loc[3],shajarian01_face_loc[0]),
                         (shajarian01_face_loc[1], shajarian01_face_loc[2]),(255,0,255),3)


imgShajarian01 = cv2.cvtColor(imgShajarian01,cv2.COLOR_RGB2BGR)
cv2.imshow('shajarian basic image', imgShajarian01)
cv2.waitKey(0)