import torch 
import cv2
import numpy as np
import winsound  # Para generar un beep en Windows

# Configura el modelo
weights_path = 'D:/Proyecto_yolo/yolov5/runs/train/exp6/weights/best.pt'  # Ruta a tus pesos entrenados
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

# Mapeo de clases a nombres
class_names = {0: "Casco", 1: "Cachucha", 2: "Tapabocas"}

# Variable para la alerta
alerta = False

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a un formato que PIL pueda manejar
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar la detección
    results = model(img)

    # Establecer el umbral de confianza
    results = results.pandas().xyxy[0]  # Convertir a pandas DataFrame
    detections = results[results['confidence'] >= 0.1]  # Filtrar detecciones por confianza

    # Dibujar las detecciones en la imagen
    for _, row in detections.iterrows():
        box = row[['xmin', 'ymin', 'xmax', 'ymax']].values
        conf = row['confidence']
        cls = int(row['class'])
        label = f'{model.names[cls]} {conf:.2f}'

        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Comprobar si la clase está en las definidas para alerta
        if cls in class_names:
            alerta = True  # Activar alerta
            alert_text = f'Alerta: {class_names[cls]} detectado!'
            cv2.putText(frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Si hay una alerta, hacer beep
    if alerta:
        winsound.Beep(1000, 500)  # Beep a 1000 Hz durante 500 ms

    # Mostrar el frame con detecciones
    cv2.imshow('Detecciones', frame)

    # Presiona 'q' para desactivar la alerta
    if cv2.waitKey(1) & 0xFF == ord('q'):
        alerta = False  # Desactivar alerta
    # Presiona 'Esc' para salir del bucle
    elif cv2.waitKey(1) & 0xFF == 27:  # 27 es el código ASCII para 'Esc'
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()