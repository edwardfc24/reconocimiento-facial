# Importamos las librerías
import numpy as np
import cv2
import pickle

cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers/face-trainner.yml")

labels = {}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

# Obtener el vídeo
capture = cv2.VideoCapture(0)
while(True):
	ret, frame = capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Habilito el el clasificador con los mismos parámetros del training
	faces = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
	# Recorro el resultado y obtengo la región de interés
	for (x, y, w, h) in faces:
		roi_gray = gray[y:y+h, x:x+w] 
		roi_color = frame[y:y+h, x:x+w]
		id, conf = recognizer.predict(roi_gray)
		if conf>=4 and conf <= 85:
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id - 1]
			color = (255, 255, 255)
			stroke = 2
			# Dibujo la etiqueta encima de la imagen
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		# Dibujo un rectángulo que enmarca el rostro
		color = (255, 0, 0) #BGR 0-255 
		stroke = 2
		final_x = x + w
		final_y = y + h
		cv2.rectangle(frame, (x, y), (final_x, final_y), color, stroke)
	# Mostrar el resultado
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
# Cerrar el proceso de la cámara y liberarlo
capture.release()
cv2.destroyAllWindows()