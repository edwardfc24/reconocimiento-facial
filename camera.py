# Importamos las librerías
import numpy as np
import cv2
# Obtener el vídeo
capture = cv2.VideoCapture(0)
while(True):
	ret, frame = capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_Lab2RGB)
	# Mostrar el resultado
	cv2.imshow('frame', frame)
	cv2.imshow('gray', gray)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
# Cerrar el proceso de la cámara y liberarlo
capture.release()
cv2.destroyAllWindows()