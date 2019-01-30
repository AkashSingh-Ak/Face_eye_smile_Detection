# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:51:50 2018

@author: AKASH SINGH THAGUNNA
"""
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def face_detect(gray, frame):
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        y_gray = gray[y:y+h, x:x+w]
        y_color = frame[y:y+h, x:x+w]
        eye = eye_cascade.detectMultiScale(y_gray, 1.1, 22)
        
        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(y_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
    
        smile = smile_cascade.detectMultiScale(y_gray, 1.7, 21)
        
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(y_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            
    return frame

video_capture = cv2.VideoCapture(0)

while(1):
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canvas = face_detect(gray, frame)
    cv2.imshow('Smile_detector', canvas)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break
video_capture.release()
cv2.destroyAllWindows()
