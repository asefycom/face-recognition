import cv2
import face_recognition
import os
import numpy as np

path = 'images_pro'
images = []
names = []

myList = os.listdir(path)
# print(myList)

for item in myList:
    curImage = cv2.imread(f'{path}/{item}')
    images.append(curImage)
    names.append(os.path.splitext(item)[0])

# print(images)
# print(names)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


known_encode_list = findEncodings(images)
# print(len(known_encode_list))
print("Encoding Completed!")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame_small = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    fame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    faces_loc = face_recognition.face_locations(frame_small)
    faces_encode = face_recognition.face_encodings(frame_small, faces_loc)

    for encodeFace, faceLoc in zip(faces_encode, faces_loc):
        matches = face_recognition.compare_faces(known_encode_list, encodeFace)
        face_distance = face_recognition.face_distance(known_encode_list, encodeFace)
        matchIndex = np.argmin(face_distance)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc

            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame,(x1,y1), (x2,y2), (255,0,255), 2)
            cv2.rectangle(frame, (x1,y2-35), (x2,y2), (255,0,255), cv2.FILLED)
            cv2.putText(frame,name,
                        (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
