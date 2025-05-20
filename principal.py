from ultralytics import YOLO
import cv2
import torch
import serial
import time
import winsound
import re
import threading

# ----------------- CONFIGURACIÓN -----------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")
modelo = YOLO("yolov8n.pt")
modelo.to(device)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

try:
    arduino = serial.Serial(port="COM6", baudrate=9600, timeout=1)
    time.sleep(2)
    print("Conectado a Arduino en COM6")
except Exception as e:
    print(f"Error al conectar con Arduino: {e}")
    arduino = None

ultimo_beep = time.time()


# ----------------- FUNCIÓN: LECTURA SENSOR -----------------


def leer_sensor():
    global ultimo_beep
    while True:
        try:
            if arduino and arduino.in_waiting > 0:
                data = arduino.readline().decode().strip()
                if data:
                    print("Sensor:", data)
                    match = re.search(r"(\d+)", data)
                    if match:
                        distancia = int(match.group(1))
                        tiempo_actual = time.time()

                        if 300 < distancia <= 200:
                            intervalo = 2
                        elif 200 < distancia <= 100:
                            intervalo = 1
                        elif 0 < distancia <= 100:
                            intervalo = 0.001
                        else:
                            intervalo = None

                        if (
                            intervalo is not None
                            and (tiempo_actual - ultimo_beep) >= intervalo
                        ):
                            winsound.Beep(1000, 300)
                            ultimo_beep = tiempo_actual
        except Exception as e:
            print(f"Error en sensor: {e}")
            break


# ----------------- INICIAR HILO PARA SENSOR -----------------

sensor_thread = threading.Thread(target=leer_sensor, daemon=True)
sensor_thread.start()

# ----------------- BUCLE PRINCIPAL DE DETECCIÓN -----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    altura, ancho, _ = frame.shape

    # Zona central de 80 cm (simulada como 50% del ancho)
    ancho_zona = int(ancho * 0.5)
    x_inicio = int((ancho - ancho_zona) / 2)
    x_fin = x_inicio + ancho_zona
    color_zona = (255, 255, 0)

    # Dibuja el área delimitada
    cv2.rectangle(frame, (x_inicio, 0), (x_fin, altura), color_zona, 2)

    resultados = modelo.predict(
        source=frame, device=0 if device == "cuda" else "cpu", conf=0.4, verbose=False
    )

    for resultado in resultados:
        for box in resultado.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            clase_id = int(box.cls[0])
            etiqueta = modelo.names[clase_id]

            # Comprobar si el objeto está en la zona central
            centro_x = (x1 + x2) // 2
            if x_inicio <= centro_x <= x_fin:
                tiempo_actual = time.time()
                if tiempo_actual - ultimo_beep >= 1:  # 1s de intervalo
                    winsound.Beep(1500, 200)
                    ultimo_beep = tiempo_actual

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{etiqueta} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    cv2.imshow("YOLOv8 + Zona de Seguridad", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
print("Finalizado correctamente.")
