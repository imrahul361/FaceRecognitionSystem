import os
import cv2
import numpy as np
import faceRecognition as fr


# Captures Images via Webcam and Performs Face Recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')  # Load saved training data

name = {0: "Name1", 1: "Name2"}


cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # Captures frame and returns boolean value and captured image
    faces_detected, gray_img = fr.faceDetection(test_img)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face detection Tutorial ', resized_img)
    cv2.waitKey(10)

    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y+w, x:x+h]
        label, confidence = face_recognizer.predict(roi_gray)  # Predicting the label of given image
        print("confidence:", confidence)
        print("label:", label)
        fr.draw_rect(test_img, face)
        predicted_name = name[label]
        if confidence < 39:
            fr.put_text(test_img, predicted_name, x, y)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Face Recognition System ', resized_img)
    if cv2.waitKey(10) == ord("q"):  # Wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows
