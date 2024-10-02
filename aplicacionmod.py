import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 entrenado desde la ruta local
model = YOLO('D:/Usuario/Desktop/Ingeniotec/yolov8/runs/detect/train/weights/best.pt')

# Usar la cámara (asegúrate de que el índice de la cámara sea correcto, en general es 0 para la cámara por defecto)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se ha abierto correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Bucle de captura de video y detección en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: no se pudo capturar el cuadro.")
        break

    # Realizar predicciones en el cuadro capturado usando YOLOv8
    results = model(frame)

    # Dibujar las cajas y las etiquetas en el cuadro original
    annotated_frame = results[0].plot()  # results[0] es el primer (y único) conjunto de predicciones

    # Mostrar el cuadro con las detecciones
    cv2.imshow("YOLOv8 - Detección en tiempo real", annotated_frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
