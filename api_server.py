# ============================================================
# VISIONAI V2
# API DO DASHBOARD
#
# Responsável por:
# - acessar a câmera
# - executar InsightFace
# - reconhecer pessoas cadastradas
# - gerar vídeo anotado
# - disponibilizar estado para o dashboard
#
# NÃO contém interface.
# A interface fica no Next.js.
# ============================================================

import os
import sys
import cv2
import time
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from insightface.app import FaceAnalysis


# ============================================================
# RAIZ DO PROJETO
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    ROOT_DIR
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

from config import (
    MODEL_NAME,
    CAMERA_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    DET_SIZE,
    FACES_DIR,
    RECOGNITION_THRESHOLD
)


# ============================================================
# COMPONENTES DO VISIONAI
# ============================================================

from camera.camera_manager import CameraManager

from face.face_database import FaceDatabase

from face.recognition_engine import RecognitionEngine


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="VisionAI V2 API",
    version="2.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# ESTADO GLOBAL
# ============================================================

camera = None

insightface = None

recognizer = None

database_manager = None

latest_frame = None

latest_result = {
    "status": "initializing",
    "environment": "Aguardando sistema",
    "people_count": 0,
    "authorized_count": 0,
    "unauthorized_count": 0,
    "faces": [],
    "camera_online": False,
    "ai_online": False,
    "fps": 0,
    "inference_ms": 0
}


lock = threading.Lock()

running = False

inference_thread = None


# ============================================================
# MÉTRICAS
# ============================================================

fps_counter = 0

fps_timer = time.perf_counter()

current_fps = 0


# ============================================================
# INTERVALO DA IA
# ============================================================

AI_INTERVAL = 0.25


# ============================================================
# INICIALIZAR SISTEMA
# ============================================================

def initialize_system():

    global camera
    global insightface
    global recognizer
    global database_manager
    global running
    global inference_thread

    print()
    print("=" * 70)
    print("              VISIONAI V2 - API")
    print("=" * 70)
    print()

    # ========================================================
    # BANCO
    # ========================================================

    print(
        "[1/3] Carregando banco de faces..."
    )

    database_manager = FaceDatabase(
        FACES_DIR
    )

    database_manager.load()

    database = (
        database_manager.database
    )

    print(
        f"[DATABASE] Pessoas: "
        f"{len(database_manager.people())}"
    )

    print(
        f"[DATABASE] Embeddings: "
        f"{database_manager.total_embeddings()}"
    )

    # ========================================================
    # RECONHECEDOR
    # ========================================================

    recognizer = RecognitionEngine(
        database=database,
        threshold=RECOGNITION_THRESHOLD
    )

    # ========================================================
    # INSIGHTFACE
    # ========================================================

    print(
        "[2/3] Carregando InsightFace..."
    )

    insightface = FaceAnalysis(
        name=MODEL_NAME,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    insightface.prepare(
        ctx_id=0,
        det_size=DET_SIZE
    )

    print(
        "[INSIGHTFACE] Online."
    )

    # ========================================================
    # CÂMERA
    # ========================================================

    print(
        "[3/3] Abrindo câmera..."
    )

    camera = CameraManager(
        camera_id=CAMERA_ID,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        buffer_size=1
    )

    camera.open()

    camera.start()

    print(
        "[CAMERA] Online."
    )

    # ========================================================
    # THREAD DA IA
    # ========================================================

    running = True

    inference_thread = threading.Thread(
        target=inference_loop,
        daemon=True
    )

    inference_thread.start()

    print()
    print("=" * 70)
    print("             VISIONAI V2 ONLINE")
    print("=" * 70)
    print()


# ============================================================
# LOOP DE INFERÊNCIA
# ============================================================

def inference_loop():

    global latest_frame
    global latest_result
    global current_fps
    global fps_counter
    global fps_timer

    last_inference = 0.0

    # Últimas detecções conhecidas.
    # O vídeo continua sendo atualizado pela câmera,
    # enquanto a IA atualiza essas informações periodicamente.

    last_faces = []

    last_inference_ms = 0.0

    while running:

        # ====================================================
        # PEGAR FRAME MAIS RECENTE
        # ====================================================

        frame, timestamp = (
            camera.get_latest_frame()
        )

        if frame is None:

            time.sleep(0.005)

            continue

        now = time.perf_counter()

        # ====================================================
        # FPS
        # ====================================================

        fps_counter += 1

        if (
            now - fps_timer
            >= 1.0
        ):

            current_fps = (
                fps_counter
            )

            fps_counter = 0

            fps_timer = now

        # ====================================================
        # VERIFICAR SE É HORA DE RODAR A IA
        # ====================================================

        should_infer = (
            now - last_inference
            >= AI_INTERVAL
        )

        # ====================================================
        # INFERÊNCIA
        # ====================================================

        if should_infer:

            last_inference = now

            start_inference = (
                time.perf_counter()
            )

            # ------------------------------------------------
            # INSIGHTFACE
            # ------------------------------------------------

            faces = insightface.get(
                frame
            )

            last_inference_ms = (
                time.perf_counter()
                -
                start_inference
            ) * 1000

            # ------------------------------------------------
            # NOVAS DETECÇÕES
            # ------------------------------------------------

            detected_faces = []

            authorized_count = 0

            unauthorized_count = 0

            # =================================================
            # PROCESSAR CADA ROSTO
            # =================================================

            for face in faces:

                # ---------------------------------------------
                # BOUNDING BOX
                # ---------------------------------------------

                bbox = (
                    face.bbox.astype(int)
                )

                x1, y1, x2, y2 = bbox

                # Garantir limites da imagem

                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    frame.shape[1] - 1,
                    x2
                )

                y2 = min(
                    frame.shape[0] - 1,
                    y2
                )

                # ---------------------------------------------
                # EMBEDDING
                # ---------------------------------------------

                embedding = (
                    face.embedding
                )

                if embedding is None:

                    identity = (
                        "desconhecido"
                    )

                    score = 0.0

                    margin = 0.0

                else:

                    result = (
                        recognizer.recognize(
                            embedding
                        )
                    )

                    identity = (
                        result["identity"]
                    )

                    score = (
                        result["score"]
                    )

                    margin = (
                        result["margin"]
                    )

                # ---------------------------------------------
                # AUTORIZAÇÃO
                # ---------------------------------------------

                authorized = (
                    identity != "desconhecido"
                )

                if authorized:

                    authorized_count += 1

                else:

                    unauthorized_count += 1

                # ---------------------------------------------
                # GUARDAR DETECÇÃO
                # ---------------------------------------------

                detected_faces.append({

                    "name":
                        identity,

                    "authorized":
                        authorized,

                    "score":
                        round(
                            float(score),
                            4
                        ),

                    "margin":
                        round(
                            float(margin),
                            4
                        ),

                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                })

            # ------------------------------------------------
            # GUARDAR ÚLTIMAS DETECÇÕES
            # ------------------------------------------------

            last_faces = detected_faces

            # =================================================
            # SITUAÇÃO DO AMBIENTE
            # =================================================

            if unauthorized_count > 0:

                environment = (
                    "ATENÇÃO - PESSOA NÃO AUTORIZADA"
                )

                status = "alert"

            elif len(faces) > 0:

                environment = (
                    "AMBIENTE SEGURO"
                )

                status = "safe"

            else:

                environment = (
                    "AMBIENTE SEM PESSOAS"
                )

                status = "safe"

            # =================================================
            # ATUALIZAR ESTADO
            # =================================================

            with lock:

                latest_result = {

                    "status":
                        status,

                    "environment":
                        environment,

                    "people_count":
                        len(faces),

                    "authorized_count":
                        authorized_count,

                    "unauthorized_count":
                        unauthorized_count,

                    "faces":
                        detected_faces,

                    "camera_online":
                        True,

                    "ai_online":
                        True,

                    "fps":
                        current_fps,

                    "inference_ms":
                        round(
                            last_inference_ms,
                            2
                        )
                }

        # ====================================================
        # CRIAR FRAME PARA EXIBIÇÃO
        #
        # O frame da câmera continua sendo atualizado.
        # As últimas caixas conhecidas são desenhadas sobre
        # o frame mais recente.
        # ====================================================

        display_frame = frame.copy()

        # ====================================================
        # DESENHAR DETECÇÕES
        # ====================================================

        for detection in last_faces:

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            identity = (
                detection["name"]
            )

            authorized = (
                detection["authorized"]
            )

            score = (
                detection["score"]
            )

            # ------------------------------------------------
            # COR
            # ------------------------------------------------

            if authorized:

                # Verde
                color = (
                    0,
                    220,
                    100
                )

                label = (
                    f"{identity.upper()} - AUTORIZADA"
                )

            else:

                # Vermelho
                color = (
                    0,
                    60,
                    255
                )

                label = (
                    "PESSOA NAO AUTORIZADA"
                )

            # =================================================
            # CAIXA
            # =================================================

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # =================================================
            # CONFIGURAÇÃO DO TEXTO
            # =================================================

            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            font_scale = 0.55

            thickness = 2

            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            # =================================================
            # POSIÇÃO DO LABEL
            # =================================================

            label_x = x1

            label_y = y1 - 10

            # Se não houver espaço acima,
            # colocar o label abaixo.

            if (
                label_y
                -
                text_height
                -
                12
                <
                0
            ):

                label_y = (
                    y2
                    +
                    text_height
                    +
                    12
                )

            # Garantir que não saia da imagem.

            if (
                label_y
                >
                display_frame.shape[0] - 5
            ):

                label_y = (
                    display_frame.shape[0]
                    - 5
                )

            # =================================================
            # FUNDO DO LABEL
            # =================================================

            background_x1 = max(
                0,
                label_x
            )

            background_y1 = max(
                0,
                label_y
                -
                text_height
                -
                10
            )

            background_x2 = min(
                display_frame.shape[1] - 1,
                label_x
                +
                text_width
                +
                16
            )

            background_y2 = min(
                display_frame.shape[0] - 1,
                label_y
                +
                4
            )

            cv2.rectangle(
                display_frame,
                (
                    background_x1,
                    background_y1
                ),
                (
                    background_x2,
                    background_y2
                ),
                color,
                -1
            )

            # =================================================
            # TEXTO
            # =================================================

            cv2.putText(
                display_frame,
                label,
                (
                    label_x + 8,
                    label_y - 5
                ),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

            # =================================================
            # SCORE
            # =================================================

            score_text = (
                f"Score: {score:.3f}"
            )

            score_y = (
                y2 + 22
            )

            if (
                score_y
                >
                display_frame.shape[0] - 5
            ):

                score_y = (
                    display_frame.shape[0]
                    - 5
                )

            cv2.putText(
                display_frame,
                score_text,
                (
                    x1,
                    score_y
                ),
                font,
                0.50,
                color,
                2,
                cv2.LINE_AA
            )

        # ====================================================
        # ATUALIZAR FRAME DO STREAM
        # ====================================================

        with lock:

            latest_frame = (
                display_frame.copy()
            )

        # ====================================================
        # PEQUENA PAUSA
        # ====================================================

        time.sleep(0.001)


# ============================================================
# STREAM DE VÍDEO
# ============================================================

def generate_video():

    while True:

        with lock:

            if latest_frame is None:

                frame = None

            else:

                frame = (
                    latest_frame.copy()
                )

        if frame is None:

            time.sleep(0.03)

            continue

        success, buffer = (
            cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    85
                ]
            )
        )

        if not success:

            continue

        frame_bytes = (
            buffer.tobytes()
        )

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(0.01)


# ============================================================
# ENDPOINT - VÍDEO
# ============================================================

@app.get(
    "/video"
)
def video():

    return StreamingResponse(
        generate_video(),
        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# ENDPOINT - STATUS
# ============================================================

@app.get(
    "/api/status"
)
def status():

    with lock:

        return latest_result.copy()


# ============================================================
# ENDPOINT - PESSOAS
# ============================================================

@app.get(
    "/api/people"
)
def people():

    return {
        "people":
            database_manager.people(),

        "total":
            len(
                database_manager.people()
            ),

        "embeddings":
            database_manager.total_embeddings()
    }


# ============================================================
# EVENTO DE INICIALIZAÇÃO
# ============================================================

@app.on_event(
    "startup"
)
def startup_event():

    initialize_system()


# ============================================================
# ENCERRAMENTO
# ============================================================

@app.on_event(
    "shutdown"
)
def shutdown_event():

    global running

    running = False

    if camera is not None:

        camera.stop()

    print(
        "[SYSTEM] VisionAI encerrado."
    )