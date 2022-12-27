# -*- coding: utf-8 -*-
import cv2
import sys
import os
#import numpy as np

if sys.version_info[:2] >= (3, 0):
    def exec_file_wrapper(fpath, g_vars, l_vars):
        with open(fpath) as f:
            code = compile(f.read(), os.path.basename(fpath), 'exec')
            exec(code, g_vars, l_vars)

cascPath = 'C:/Users/sara/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(30, 30),
        flags=cv2.FONT_HERSHEY_SIMPLEX
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 
                'RECOGNIZING YOU FACE:', 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
  
 #black screen

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
         
        

    # Display the resulting frame
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()