import cv2
import os
import mediapipe as mp

# Inicialización de detección de rostros con MediaPipe
mp_face_detection = mp.solutions.face_detection
LABELS = ["despejada", "oculta"]

# Leer el modelo previamente entrenado (LBPH)
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("rostro_model.xml")

# Captura de video desde la cámara
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Definir margen adicional para ampliar el cuadro alrededor del rostro
MARGIN = 50  # Cambia este valor para aumentar o disminuir el tamaño del recuadro

# Detección de rostros con MediaPipe
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Espejar la imagen para que se vea como un espejo
        frame = cv2.flip(frame, 1)
        
        # Obtener dimensiones del cuadro
        height, width, _ = frame.shape
        
        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        # Si se detectan caras
        if results.detections:
            for detection in results.detections:
                # Obtener coordenadas del cuadro delimitador
                bbox = detection.location_data.relative_bounding_box
                xmin = int(bbox.xmin * width)
                ymin = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Aumentar el tamaño del recuadro con el margen
                xmin = max(0, xmin - MARGIN)  # Asegurar que xmin no sea negativo
                ymin = max(0, ymin - MARGIN)  # Asegurar que ymin no sea negativo
                xmax = min(width, xmin + w + 1.5 * MARGIN)  # Limitar a los bordes de la imagen
                ymax = min(height, ymin + h + 1.5 * MARGIN)  # Limitar a los bordes de la imagen

                # Extraer la región de la cara con el margen adicional
                face_image = frame[ymin:ymax, xmin:xmax]
                if face_image.size == 0:  # Si el tamaño de la imagen es cero, omitir
                    continue

                # Convertir a escala de grises
                face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

                # Redimensionar la imagen al tamaño esperado por el modelo
                face_image_resized = cv2.resize(face_image_gray, (244, 244), interpolation=cv2.INTER_CUBIC)

                # Realizar la predicción usando el modelo entrenado
                result = face_mask.predict(face_image_resized)
                label, confidence = result

                # Mostrar el resultado si la confianza es suficientemente alta
                if confidence < 150:  # Ajusta este umbral según los resultados de tus pruebas
                    label_text = LABELS[label]
                    color = (0, 255, 0) if label_text == "despejada" else (0, 0, 255)

                    # Mostrar el nombre del estado (despejada/oculta)
                    cv2.putText(frame, label_text, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    # Dibujar el recuadro alrededor del rostro con el margen ampliado
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Mostrar el video con las predicciones
        cv2.imshow("Frame", frame)

        # Salir si se presiona la tecla 'Esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Liberar la captura de video y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
