"""
app.py — YOLOv11 & YOLOv26 Multi-Device Explorer (v2)
=============================================
ARQUITECTURA:
- Hilo de captura: Lee frames de la webcam del servidor (modo SERVER)
                   o espera frames enviados por un cliente via POST (modo CLIENT)
- Hilo de procesamiento: Aplica YOLO sobre el raw_frame y guarda el resultado anotado
- Flask: Sirve la página web, el stream MJPEG, y la API REST de control

Los diccionarios STATE y CONFIG son compartidos entre hilos.
Se usa STATE_LOCK para proteger las escrituras críticas (raw_frame, current_frame)
y evitar race conditions.
"""

import os
import time
import uuid
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- LÍNEA DE COMUNICACIÓN ENTRE HILOS ---
# Lock para proteger escrituras concurrentes al estado compartido.
# Esto previene "race conditions" donde dos hilos escriben a la vez.
STATE_LOCK = threading.Lock()

# --- ESTADO GLOBAL DE LA APLICACIÓN ---
CONFIG = {
    "model_name": "yolo11n-seg.pt",
    "conf_threshold": 0.45,
    "current_model": None,
    "model_loading": False,   # True mientras se carga un modelo nuevo
    "load_lock": threading.Lock()
}

STATE = {
    "fps": 0,
    "inference_ms": 0,        # Tiempo solo de inferencia (sin encoding)
    "last_detections": [],
    "camera_mode": "server",  # 'server' o 'client'
    "active_token": None,
    "token_last_seen": 0,
    "current_frame": None,    # Frame anotado listo para el stream MJPEG
    "raw_frame": None         # Frame sin procesar (entrada a YOLO)
}

# ─────────────────────────────────────────
# COLORES POR CATEGORÍA COCO
# ─────────────────────────────────────────
# Los colores están en formato BGR (Blue, Green, Red) que es como OpenCV los usa.
CATEGORY_COLORS = {
    "person":      (0, 220, 0),      # Verde brillante
    "vehicle":     (0, 60, 255),     # Rojo
    "animal":      (0, 220, 220),    # Amarillo
    "electronic":  (255, 100, 0),    # Azul
    "food":        (0, 160, 255),    # Naranja
    "furniture":   (160, 160, 160),  # Gris
    "sport":       (220, 0, 220),    # Violeta
    "outdoor":     (0, 180, 130),    # Verde azulado
    "other":       (200, 200, 200),  # Gris claro
}

CLASS_TO_CATEGORY = {
    # Personas
    "person": "person",
    # Vehículos
    "bicycle": "vehicle", "car": "vehicle", "motorcycle": "vehicle",
    "airplane": "vehicle", "bus": "vehicle", "train": "vehicle",
    "truck": "vehicle", "boat": "vehicle",
    # Animales
    "bird": "animal", "cat": "animal", "dog": "animal", "horse": "animal",
    "sheep": "animal", "cow": "animal", "elephant": "animal", "bear": "animal",
    "zebra": "animal", "giraffe": "animal",
    # Electrónica
    "tv": "electronic", "laptop": "electronic", "mouse": "electronic",
    "remote": "electronic", "keyboard": "electronic", "cell phone": "electronic",
    # Alimentos
    "banana": "food", "apple": "food", "sandwich": "food", "orange": "food",
    "broccoli": "food", "carrot": "food", "hot dog": "food", "pizza": "food",
    "donut": "food", "cake": "food",
    # Muebles
    "chair": "furniture", "couch": "furniture", "bed": "furniture",
    "dining table": "furniture", "toilet": "furniture", "potted plant": "furniture",
    # Deportes
    "frisbee": "sport", "skis": "sport", "snowboard": "sport",
    "sports ball": "sport", "kite": "sport", "baseball bat": "sport",
    "baseball glove": "sport", "skateboard": "sport", "surfboard": "sport",
    "tennis racket": "sport",
    # Objetos exteriores
    "traffic light": "outdoor", "fire hydrant": "outdoor", "stop sign": "outdoor",
    "parking meter": "outdoor", "bench": "outdoor",
}

def get_color(cls_name):
    """Devuelve el color BGR asociado a la categoría del objeto detectado."""
    category = CLASS_TO_CATEGORY.get(cls_name, "other")
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"])


# ─────────────────────────────────────────
# CARGA DEL MODELO
# ─────────────────────────────────────────
def load_model(name):
    """
    Carga el modelo YOLO indicado.
    Se ejecuta en un hilo separado para no bloquear Flask durante el cambio.
    Usa un Lock (load_lock) para que solo un hilo cargue el modelo a la vez.
    """
    with CONFIG["load_lock"]:
        CONFIG["model_loading"] = True
        print(f"📦 Cargando modelo {name}...")
        try:
            CONFIG["current_model"] = YOLO(name)
            CONFIG["model_name"] = name
            print(f"✅ Modelo {name} listo.")
        except Exception as e:
            print(f"❌ Error cargando modelo {name}: {e}")
        finally:
            CONFIG["model_loading"] = False

# Carga inicial del modelo al arrancar
load_model(CONFIG["model_name"])


# ─────────────────────────────────────────
# HILO 1: CAPTURA DE LA CÁMARA DEL SERVIDOR
# ─────────────────────────────────────────
def server_camera_thread():
    """
    Intenta abrir la webcam local (índice 0).
    En Windows, CAP_DSHOW evita el error 'obsensor/out of range'.
    Si no hay webcam, el servidor queda esperando frames de clientes.
    """
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print("⚠️  AVISO: No se detectó webcam local (índice 0).")
        print("   Modo 'Cámara del Servidor' no disponible.")
        print("   Usa el botón '📷 Usar mi cámara' desde tu celular.")
        # Si no hay webcam, esperamos frames del cliente
        STATE["camera_mode"] = "client"

    while True:
        if STATE["camera_mode"] == "server" and cap.isOpened():
            success, frame = cap.read()
            if success:
                with STATE_LOCK:
                    STATE["raw_frame"] = frame
            else:
                time.sleep(0.05)
        else:
            # Si hay una imagen 'sample.jpg', la cargamos una vez para pruebas
            if STATE["raw_frame"] is None and os.path.exists("sample.jpg"):
                sample = cv2.imread("sample.jpg")
                if sample is not None:
                    with STATE_LOCK:
                        STATE["raw_frame"] = sample
            time.sleep(0.05)

    cap.release()


# ─────────────────────────────────────────
# HILO 2: PROCESAMIENTO YOLO
# ─────────────────────────────────────────
def processing_thread():
    """
    Toma el raw_frame actual, aplica YOLO, dibuja las cajas y guarda el resultado.
    Gestiona el timeout del token de cámara cliente.
    """
    while True:
        with STATE_LOCK:
            raw = STATE["raw_frame"]

        if raw is None:
            time.sleep(0.02)
            continue

        # Verificar si el token del cliente expiró (sin frames por más de 10s)
        if STATE["camera_mode"] == "client":
            if STATE["active_token"] and (time.time() - STATE["token_last_seen"] > 10):
                print("⏰ Token expirado — volviendo a cámara del servidor.")
                STATE["camera_mode"] = "server"
                STATE["active_token"] = None

        # Guard: si el modelo todavía está cargando, esperamos
        model = CONFIG["current_model"]
        if model is None or CONFIG["model_loading"]:
            time.sleep(0.05)
            continue

        frame = raw.copy()

        # --- Inferencia YOLO ---
        # Solo medimos el tiempo de inferencia pura (no el encoding del frame)
        t_inf_start = time.time()
        results = model(frame, conf=CONFIG["conf_threshold"], verbose=False, device='cpu')[0]
        inference_ms = int((time.time() - t_inf_start) * 1000)

        # --- Dibujar detecciones ---
        detections = []
        
        # Overlay para máscaras (segmentación)
        mask_overlay = frame.copy()
        has_masks = hasattr(results, 'masks') and results.masks is not None

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            color = get_color(cls_name)

            # Dibujar Máscara si existe
            if has_masks:
                # results.masks.xy es una lista de polígonos
                # Cada polígono es un array numpy de (N, 2)
                polygon = results.masks.xy[i].astype(np.int32)
                if len(polygon) > 0:
                    cv2.fillPoly(mask_overlay, [polygon], color)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Etiqueta con fondo de color
            label = f"{cls_name} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_y = max(y1, lh + 8)
            cv2.rectangle(frame, (x1, label_y - lh - 8), (x1 + lw + 4, label_y), color, -1)
            cv2.putText(frame, label, (x1 + 2, label_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            detections.append({
                "class": cls_name,
                "conf": round(conf, 2),
                "category": CLASS_TO_CATEGORY.get(cls_name, "other")
            })

        # Mezclar máscaras con el frame original (transparencia 40%)
        if has_masks:
            alpha = 0.4
            cv2.addWeighted(mask_overlay, alpha, frame, 1 - alpha, 0, frame)

        # FPS real = 1000ms / tiempo de inferencia en ms
        fps = int(1000 / inference_ms) if inference_ms > 0 else 0

        # Codificar el frame procesado a JPEG para el stream
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Escritura protegida por lock
        with STATE_LOCK:
            STATE["current_frame"] = buffer.tobytes()
            STATE["fps"] = fps
            STATE["inference_ms"] = inference_ms
            STATE["last_detections"] = detections


# Iniciar hilos como daemon (se cierran cuando el proceso principal termina)
threading.Thread(target=server_camera_thread, daemon=True).start()
threading.Thread(target=processing_thread, daemon=True).start()


# ─────────────────────────────────────────
# RUTAS FLASK
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    """Generador MJPEG: envía el frame actual a todos los clientes conectados."""
    while True:
        with STATE_LOCK:
            frame = STATE["current_frame"]

        if frame:
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except GeneratorExit:
                # El cliente se desconectó — limpieza segura
                return
        time.sleep(0.04)  # Cap a ~25 fps de stream


@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    with STATE_LOCK:
        detections = list(STATE["last_detections"])
        fps = STATE["fps"]
        inference_ms = STATE["inference_ms"]
        current_frame_exists = STATE["current_frame"] is not None

    return jsonify({
        "fps": fps,
        "inference_ms": inference_ms,
        "model": CONFIG["model_name"],
        "model_loading": CONFIG["model_loading"],
        "camera_mode": STATE["camera_mode"],
        "detections": detections,
        "client_locked": STATE["active_token"] is not None,
        "has_frame": current_frame_exists
    })


@app.route('/config', methods=['POST'])
def update_config():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    if "conf_threshold" in data:
        CONFIG["conf_threshold"] = float(data["conf_threshold"])

    if "model" in data and data["model"] != CONFIG["model_name"] and not CONFIG["model_loading"]:
        # Cargar el nuevo modelo en un hilo separado para no bloquear la respuesta
        threading.Thread(target=load_model, args=(data["model"],), daemon=True).start()

    return jsonify({"status": "ok"})


@app.route('/claim_camera', methods=['POST'])
def claim_camera():
    """
    Un cliente solicita convertirse en la fuente de video.
    Solo puede haber un cliente activo a la vez.
    """
    if STATE["active_token"] is None:
        token = str(uuid.uuid4())
        STATE["active_token"] = token
        STATE["camera_mode"] = "client"
        STATE["token_last_seen"] = time.time()
        return jsonify({"token": token})
    return jsonify({"error": "Cámara ocupada por otro dispositivo"}), 409


@app.route('/release_camera', methods=['POST'])
def release_camera():
    """El cliente activo libera el control de la cámara."""
    token = request.headers.get('X-Camera-Token')
    if token and token == STATE["active_token"]:
        STATE["active_token"] = None
        STATE["camera_mode"] = "server"
        return jsonify({"status": "released"})
    return jsonify({"error": "Token inválido o sin permisos"}), 403


@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """
    Recibe un frame JPEG crudo desde el cliente activo.
    Solo acepta requests con el token correcto en el header.
    """
    token = request.headers.get('X-Camera-Token')
    if not token or token != STATE["active_token"]:
        return jsonify({"error": "Sin autorización"}), 401

    STATE["token_last_seen"] = time.time()
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        with STATE_LOCK:
            STATE["raw_frame"] = img
        return jsonify({"status": "ok"})

    return jsonify({"error": "Frame inválido"}), 400


# ─────────────────────────────────────────
# ARRANQUE DEL SERVIDOR
# ─────────────────────────────────────────
if __name__ == '__main__':
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()

    print("\n" + "=" * 52)
    print("🚀  YOLOv11 & YOLOv26 Multi-Device Explorer")
    print("=" * 52)
    print(f"🔗  Local:    http://localhost:5001")
    print(f"📱  Red WiFi: http://{local_ip}:5001")
    print("=" * 52)
    print("   Abri esa URL en cualquier celular de tu red")
    print("=" * 52 + "\n")

    try:
        app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)
    except Exception as e:
        print(f"❌ Error al iniciar el servidor: {e}")
