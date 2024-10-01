import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

# Configura el modelo YOLO
weights_path = 'D:/Proyecto_yolo/yolov5/runs/train/exp6/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Inicializa MTCNN para detección de rostros
mtcnn = MTCNN(keep_all=True)

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a un formato que PIL pueda manejar
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostros con MTCNN
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            # Ampliar el cuadro de detección 20 píxeles a la izquierda y a la derecha
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1 - 30)
            x2 = min(frame.shape[1], x2 + 30)
            y1 = max(0, y1)
            y2 = min(frame.shape[0], y2)

            # Extraer el rostro
            face = frame[y1:y2, x1:x2]

            # Redimensionar la imagen a 224x224
            face_resized = cv2.resize(face, (224, 224))

            # Realizar la detección con YOLOv5
            results = model(face_resized)

            # Establecer el umbral de confianza
            results = results.pandas().xyxy[0]
            detections = results[results['confidence'] >= 0.1]

            # Dibujar las detecciones en el rostro
            for _, row in detections.iterrows():
                conf = row['confidence']
                cls = int(row['class'])
                label = f'{model.names[cls]} {conf:.2f}'

                cv2.rectangle(face_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(face_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mostrar el frame con detecciones
    cv2.imshow('Detecciones', face_resized)

    # Presiona 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
