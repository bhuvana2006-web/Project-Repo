import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture=cv2.VideoCapture(0)

pm_image=face_recognition.load_image_file("photos/pm.jpeg")
pm_encoding = face_recognition.face_encodings(pm_image)[0]

cm_image=face_recognition.load_image_file("photos/cm.jpeg")
cm_encoding = face_recognition.face_encodings(cm_image)[0]

depcm_image=face_recognition.load_image_file("photos/depcm.jpeg")
depcm_encoding = face_recognition.face_encodings(depcm_image)[0]


known_face_encoding = [
pm_encoding,
cm_encoding,
depcm_encoding
]

known_faces_names = [
"Sri Narendra Modi",
"Sri N Chandra Babu Naidu",
"Sri K Pawan Kalyan"
]

politicians=known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s= True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f= open("current_date+'.csv','w+',newline=")
Inwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                font=cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText=(10,100)
                fontScale      =1.5
                fontColor      =(255,0,0)
                thickness      =3
                lineType       =2

                cv2.putText(frame,name+'present',
                  bottomLeftCornerOfText,
                  font,
                  fontScale,
                  fontColor,
                  thickness,
                  lineType)

                if name in politicians:
                    politicians.remove(name)
                    # print(politicians)
                    current_time=now.strftime("%H-%M-%S")
                    Inwriter.writerow([name,current_time])
        cv2.imshow("ATTENDANCE SYSTEM",frame)
        if cv2.waitkey(1)& 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()
f.close()