import cv2
import torch
from facenet_pytorch import MTCNN

# Ruta a tu modelo reentrenado
weights_path = 'D:/Usuario/Desktop/Ingeniotec/prueba/yolov5/runs/train/exp4/weights/best.pt'

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

# Inicializar MTCNN
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Iniciar la captura de video
cap = cv2.VideoCapture(0)  # Cambia '0' por la ruta de tu video si es necesario

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede capturar el video.")
        break

    # Detectar rostros
    boxes, _ = mtcnn.detect(frame)

    # Si se detectan rostros
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]

            # Aumentar el rango de detección
            x1 = max(0, x1 - 30)
            y1 = max(0, y1 - 30)
            x2 = min(frame.shape[1], x2 + 30)
            y2 = min(frame.shape[0], y2 + 30)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar un rectángulo alrededor del rostro

            # Recortar la región del rostro
            face = frame[y1:y2, x1:x2]

            # Realizar la detección con YOLOv5 en el rostro recortado
            results = model(face)

            # Obtener las detecciones en formato DataFrame
            detections = results.pandas().xyxy[0]

            # Dibujar las detecciones en el rostro
            for index, row in detections.iterrows():
                x1_det, y1_det, x2_det, y2_det, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
                # Ajustar las coordenadas de la detección al marco original
                x1_det += x1
                x2_det += x1
                y1_det += y1
                y2_det += y1
                
                cv2.rectangle(frame, (int(x1_det), int(y1_det)), (int(x2_det), int(y2_det)), (255, 0, 0), 2)
                cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1_det), int(y1_det)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mostrar el fotograma con las detecciones
    cv2.imshow('Face Detection with YOLOv5', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
