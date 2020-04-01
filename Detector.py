from mtcnn.mtcnn import MTCNN
import cv2
import sys
import time
from PIL import Image
import numpy as np
from numpy import asarray
import tensorflow as tf
from scipy import spatial
import os
class Detector:
    #Initializing the MTCNN and loading the facenet
    def __init__(self):
        self.detector = MTCNN()
        self.model = tf.keras.models.load_model('facenet_keras.h5')

    #A function that detects, aligns and recognizes faces
    def detect_and_recognize(self, img, encarr, threshold=0.95, required_size=(160, 160)):
        image = img
        pixels = asarray(image)
        #Detection
        results = self.detector.detect_faces(pixels)
        for i in range(len(results)):
            #Getting the landmarks
            landmarks = results[i]['keypoints']
            #Finding the angle made by the line connecting the eyes with the horizontal
            angle = find_angle(landmarks['left_eye'][0], landmarks['left_eye'][1], landmarks['right_eye'][0], landmarks['right_eye'][1])
            x, y = [], []
            for _, point in landmarks.items():
                x.append(point[0])
                y.append(point[1])
            #Rotating the image to achieve proper alignment
            rotation_matrix = cv2.getRotationMatrix2D((np.mean(x), np.mean(y)), angle, 1)
            pixels = cv2.warpAffine(pixels, rotation_matrix, (pixels.shape[0], pixels.shape[1]))
            #Getting the co-ordinates of the detected face
            x1, y1, w, h = results[i]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + w, y1 + h
            #Displaying the properly aligned face
            face = pixels[y1:y2, x1:x2]
            img = Image.fromarray(face)
            img = img.resize(required_size)
            face_array = asarray(img)
            cv2.imshow('roi', face_array)
            #Recognition
            enc = self.get_encoding(face_array)
            name = self.compare_encodings(encarr, enc)
            #Draw rectangle around the detected face and display the recognized person's name
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(image, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Faces Detected", image)

    #A function to extract the face from an image containing a single face
    def extract_face_for_encarr(self, img, required_size=(160, 160)):
        image = img
        pixels = asarray(image)
        results = self.detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0))
        face = pixels[y1:y2, x1:x2]
        img = Image.fromarray(face)
        img = img.resize(required_size)
        face_array = asarray(img)
        return face_array

    #A function to get the 128 value vector face encoding using FaceNet
    def get_encoding(self, img):
        image = asarray(img)
        image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
        face_pixels = image.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = np.expand_dims(face_pixels, axis=0)
        yhat = self.model.predict(samples)
        return yhat[0]

    #A function to compare the face encoding with the dictionary of encodings and return the name of the closest match
    #Cosine distance is used for the comparison
    def compare_encodings(self, enc_arr, enc):
        min_dist = sys.maxsize
        identity = None
        if enc is None:
            return None
        for (name, db_enc) in enc_arr.items():
            if db_enc is not None:
                dist = spatial.distance.cosine(db_enc, enc)
                # print(dist, name)
                if dist < min_dist:
                    min_dist = dist
                    identity = name
        return identity

    #A function to generate a named list of face encodings of all the images in the People folder
    #This function returns a dictionary of {name: encoding} key-value pairs
    def generate_encarr(self):
        encarr={}
        directory = os.fsencode("./People")
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            img = cv2.imread("./People/"+filename)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = asarray(img)
            face = self.extract_face_for_encarr(img)
            face = Image.fromarray(face)
            enc = self.get_encoding(face)
            encarr[(str(file).split('.')[0]).split("'")[1]] = enc
        return encarr

def find_angle(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(slope))

#Detection and recognition
n = Detector()
#Generating the dictionary of encodings
encarr = n.generate_encarr()
#Opening the webcam
# vid_file = 'test_vid.mp4'
vid = cv2.VideoCapture(0)
time.sleep(2)
#Detecting, recognizing and displaying every frame in real time
while True:
    ret, img = vid.read()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = asarray(img)
    n.detect_and_recognize(img,encarr)
    key = cv2.waitKey(30) & 0xFF
    #Exiting if q is pressed by the user
    if key == ord("q"):
        break

