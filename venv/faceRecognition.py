import cv2
import os
import numpy as np


# For given Image function returns rectangle for face Detection
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier(
        '/HaarCascade/haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=6)

    return faces, gray_img


# Given a directory below function and returns face alongwith its Label/ID
def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('.'):
                # Skipping files that startwith .(hidden/System files)
                print("Skipping system file")
                continue

            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print("img path: ", img_path)
            print("id: ", id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect, gray_img = faceDetection(test_img)  # load each images

            if len(faces_rect) != 1:  # Assuming only Single person to be fed to Classifier
                continue
            (x, y, w, h) = faces_rect[0]
            # Cropping Region of Interest i.e. face area from grayscale images
            roi_gray = gray_img[y:y + w, x:x + h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID


# Train HAAR Classifier
def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer


# Draws bounding boxes around detected face in image
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)


# Print the Detected Person Name in Image
def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 0, 0), 6)
