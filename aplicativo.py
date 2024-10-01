import cv2
import torch

# Ruta a tu modelo reentrenado
weights_path = 'D:/Usuario/Desktop/Ingeniotec/prueba/yolov5/runs/train/exp4/weights/best.pt'

# Cargar el modelo reentrenado
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

# Iniciar la captura de video
cap = cv2.VideoCapture(0)  # Cambia '0' por la ruta de tu video si es necesario

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede capturar el video.")
        break

    # Convertir el fotograma de BGR a RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar la detección
    results = model(img)

    # Obtener las detecciones en formato DataFrame
    detections = results.pandas().xyxy[0]

    # Dibujar las detecciones en el fotograma
    for index, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mostrar el fotograma con las detecciones
    cv2.imshow('YOLOv5 Detections', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
