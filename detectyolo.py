import torch
import cv2
from PIL import Image
import numpy as np

# Configura el modelo
weights_path = 'D:/Proyecto_yolo/yolov5/runs/train/exp6/weights/best.pt'  # Ruta a tus pesos entrenados
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Cargar la imagen
img_path = r'D:/Proyecto_yolo/yolov5/yolo/images/validation/normal_79.jpg'
img = Image.open(img_path)

# Realizar la detección
results = model(img)

# Establecer el umbral de confianza
results = results.pandas().xyxy[0]  # Convertir a pandas DataFrame
detections = results[results['confidence'] >= 0.1]  # Filtrar detecciones por confianza

# Convertir la imagen a un formato que OpenCV pueda manejar
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Dibujar las detecciones en la imagen
for _, row in detections.iterrows():
    print("pasé")
    box = row[['xmin', 'ymin', 'xmax', 'ymax']].values
    conf = row['confidence']
    cls = int(row['class'])
    label = f'{model.names[cls]} {conf:.2f}'
    
    cv2.rectangle(img_cv, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.putText(img_cv, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Mostrar la imagen con detecciones
cv2.imshow('Detecciones', img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
