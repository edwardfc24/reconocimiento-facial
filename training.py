import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("jpg") or file.endswith("jpeg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace("-", " ").lower()
			id = current_id
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			pil_image = Image.open(path).convert("L")
			size = (600, 600)
			image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(image, "uint8")
			faces = cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=3)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id)

with open("pickles/face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")