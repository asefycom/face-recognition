import cv2
import face_recognition

shajarian01_file = 'images/shajarian01.jpg'

imgShajarian01 = face_recognition.load_image_file(shajarian01_file)
# imgShajarian01CV = cv2.imread(shajarian01_file)
#
# print(imgShajarian01)
# print('-----------')
# print(imgShajarian01CV)

imgShajarian01 = cv2.cvtColor(imgShajarian01,cv2.COLOR_RGB2BGR)
cv2.imshow('shajarian basic image', imgShajarian01)
cv2.waitKey(0)