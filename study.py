# region Imports
import cv2
import face_recognition
# endregion


# region Training Image 1
shajarian01_file = 'images/shajarian01.jpg'
imgShajarian01 = face_recognition.load_image_file(shajarian01_file)
shajarian01_face_loc = face_recognition.face_locations(imgShajarian01)[0]
print(shajarian01_face_loc)
face01_rec = cv2.rectangle(imgShajarian01, (shajarian01_face_loc[3],shajarian01_face_loc[0]),
                           (shajarian01_face_loc[1], shajarian01_face_loc[2]),(255,0,255),3)
shajarian_face_code = face_recognition.face_encodings(imgShajarian01)[0]
# endregion


# region Training Image 2
takhti_file = 'images/takhti.jpg'
imgTakhti = face_recognition.load_image_file(takhti_file)
takhti_face_loc = face_recognition.face_locations(imgTakhti)[0]
print(takhti_face_loc)
face01_rec = cv2.rectangle(imgTakhti, (takhti_face_loc[3],takhti_face_loc[0]),
                           (takhti_face_loc[1], takhti_face_loc[2]),(255,0,255),3)
takhti_face_code = face_recognition.face_encodings(imgTakhti)[0]
# endregion

# region Test Image
Test_file = 'images/shajarian02.jpg'
imgTest = face_recognition.load_image_file(Test_file)
test_face_loc = face_recognition.face_locations(imgTest)[0]
print(test_face_loc)
test_face_rec = cv2.rectangle(imgTest, (test_face_loc[3],test_face_loc[0]),
                           (test_face_loc[1], test_face_loc[2]),(255,0,255),3)
test_face_code = face_recognition.face_encodings(imgTest)[0]
# endregion


# region Face Recognition
results = face_recognition.compare_faces([shajarian_face_code, takhti_face_code], test_face_code)
distances = face_recognition.face_distance([shajarian_face_code, takhti_face_code], test_face_code)
print(results, distances)
# endregion


# region Show Images
imgShajarian01 = cv2.cvtColor(imgShajarian01,cv2.COLOR_RGB2BGR)
imgTakhti = cv2.cvtColor(imgTakhti,cv2.COLOR_RGB2BGR)
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)
cv2.putText(imgTest, f'{results[0]} {round(distances[0],2)}',
            (0,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255),2)
cv2.imshow('Shajarian Training image', imgShajarian01)
cv2.imshow('Takhti Training image', imgTakhti)
cv2.imshow('Test image', imgTest)
cv2.waitKey(0)
# endregion