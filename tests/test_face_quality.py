import sys
import os
import cv2
import time
import numpy as np


# ============================================================
# RAIZ DO PROJETO
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    ROOT_DIR
)


# ============================================================
# BIBLIOTECAS
# ============================================================

from insightface.app import FaceAnalysis

from config import (
    MODEL_NAME,
    CAMERA_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    DET_SIZE
)

from face.face_quality import (
    FaceQuality
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

quality_checker = FaceQuality()


# ============================================================
# INSIGHTFACE
# ============================================================

print()
print("=" * 70)
print("       VISIONAI V2 - FACE QUALITY TEST")
print("=" * 70)
print()

print(
    "Carregando InsightFace..."
)


start = time.perf_counter()


app = FaceAnalysis(
    name=MODEL_NAME,
    providers=[
        "CPUExecutionProvider"
    ]
)


app.prepare(
    ctx_id=0,
    det_size=DET_SIZE
)


elapsed = (
    time.perf_counter()
    -
    start
)


print(
    f"InsightFace carregado em "
    f"{elapsed:.2f} segundos."
)


# ============================================================
# CÂMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_ID
)


camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)


camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    print(
        "ERRO: câmera não abriu."
    )

    raise SystemExit


print()
print("Câmera iniciada.")
print()
print("Q = sair")
print()


# ============================================================
# FPS
# ============================================================

fps_counter = 0

fps_timer = time.time()

fps = 0


# ============================================================
# LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # CAPTURAR FRAME
    # --------------------------------------------------------

    ret, frame = camera.read()


    if not ret:

        continue


    # --------------------------------------------------------
    # REDUZIR IMAGEM PARA O INSIGHTFACE
    # --------------------------------------------------------

    small_frame = cv2.resize(
        frame,
        DET_SIZE,
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # DETECTAR FACES
    # --------------------------------------------------------

    start_inference = (
        time.perf_counter()
    )


    faces = app.get(
        small_frame
    )


    inference_ms = (
        time.perf_counter()
        -
        start_inference
    ) * 1000


    # --------------------------------------------------------
    # ESCALA ENTRE 320x320 E 640x480
    # --------------------------------------------------------

    scale_x = (
        frame.shape[1]
        /
        DET_SIZE[0]
    )


    scale_y = (
        frame.shape[0]
        /
        DET_SIZE[1]
    )


    # ========================================================
    # ANALISAR CADA ROSTO
    # ========================================================

    for index, face in enumerate(
        faces
    ):

        # ----------------------------------------------------
        # COORDENADAS DA FACE NO FRAME DO INSIGHTFACE
        # ----------------------------------------------------

        x1, y1, x2, y2 = (
            face.bbox.astype(float)
        )


        # ----------------------------------------------------
        # CONVERTER PARA O FRAME ORIGINAL
        # ----------------------------------------------------

        x1_original = int(
            x1 * scale_x
        )


        x2_original = int(
            x2 * scale_x
        )


        y1_original = int(
            y1 * scale_y
        )


        y2_original = int(
            y2 * scale_y
        )


        # ====================================================
        # FACE PROXY
        # ====================================================

        class FaceProxy:

            def __init__(
                self,
                bbox,
                det_score
            ):

                self.bbox = np.array(
                    bbox,
                    dtype=np.float32
                )

                self.det_score = float(
                    det_score
                )


        face_original = FaceProxy(
            [
                x1_original,
                y1_original,
                x2_original,
                y2_original
            ],
            face.det_score
        )


        # ====================================================
        # AVALIAÇÃO DA QUALIDADE
        # ====================================================

        quality = (
            quality_checker.evaluate(
                frame,
                face_original
            )
        )


        # ====================================================
        # STATUS
        # ====================================================

        if quality["quality_ok"]:

            color = (
                0,
                255,
                0
            )

            status = (
                "QUALIDADE OK"
            )

        else:

            color = (
                0,
                165,
                255
            )

            status = (
                "QUALIDADE BAIXA"
            )


        # ====================================================
        # CAIXA DO ROSTO
        # ====================================================

        cv2.rectangle(
            frame,
            (
                x1_original,
                y1_original
            ),
            (
                x2_original,
                y2_original
            ),
            color,
            2
        )


        # ====================================================
        # STATUS PRINCIPAL
        # ====================================================

        cv2.putText(
            frame,
            status,
            (
                x1_original,
                max(
                    y1_original - 10,
                    25
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # SCORE DE DETECÇÃO
        # ====================================================

        cv2.putText(
            frame,
            (
                f"Det: "
                f"{quality['detection_score']:.2f}"
            ),
            (
                x1_original,
                min(
                    y2_original + 20,
                    frame.shape[0] - 50
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # BRILHO
        # ====================================================

        cv2.putText(
            frame,
            (
                f"Brilho: "
                f"{quality['brightness']:.1f}"
            ),
            (
                x1_original,
                min(
                    y2_original + 38,
                    frame.shape[0] - 30
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # NITIDEZ
        # ====================================================

        cv2.putText(
            frame,
            (
                f"Nitidez: "
                f"{quality['sharpness']:.1f}"
            ),
            (
                x1_original,
                min(
                    y2_original + 56,
                    frame.shape[0] - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # TAMANHO DO ROSTO
        # ====================================================

        cv2.putText(
            frame,
            (
                f"Tamanho: "
                f"{quality['width']}x"
                f"{quality['height']}"
            ),
            (
                x1_original,
                max(
                    y1_original - 30,
                    45
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # FPS
    # ========================================================

    fps_counter += 1


    now = time.time()


    if (
        now
        -
        fps_timer
        >= 1
    ):

        fps = fps_counter

        fps_counter = 0

        fps_timer = now


    # ========================================================
    # PAINEL DE INFORMAÇÕES
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (390, 105),
        (0, 0, 0),
        -1
    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # FACES
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # TEMPO DE INFERÊNCIA
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"InsightFace: "
        f"{inference_ms:.1f} ms",
        (20, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # MOSTRAR
    # ========================================================

    cv2.imshow(
        "VisionAI V2 - Face Quality",
        frame
    )


    # ========================================================
    # TECLADO
    # ========================================================

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if key == ord("q"):

        break


# ============================================================
# ENCERRAMENTO
# ============================================================

camera.release()

cv2.destroyAllWindows()


print()
print("=" * 70)
print("FACE QUALITY TEST ENCERRADO")
print("=" * 70)
print()