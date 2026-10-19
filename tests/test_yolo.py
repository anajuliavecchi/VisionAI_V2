# ============================================================
# VISIONAI V2
# TESTE YOLO + CÂMERA
# ============================================================

import cv2
import time

from ultralytics import YOLO

from config import (
    CAMERA_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_BUFFER_SIZE
)

from camera.camera_manager import (
    CameraManager
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.45

IMAGE_SIZE = 416

DURACAO_TESTE = 30


# ============================================================
# INÍCIO
# ============================================================

print()
print("=" * 60)
print("             VISIONAI V2 | TESTE YOLO")
print("=" * 60)
print()

print(
    "Este teste mede o impacto do YOLO."
)

print(
    "InsightFace NÃO será utilizado neste teste."
)

print()


# ============================================================
# CARREGAR YOLO
# ============================================================

print(
    "[YOLO] Carregando modelo..."
)

inicio_modelo = time.perf_counter()


try:

    model = YOLO(
        MODEL_PATH
    )


except Exception as erro:

    print()
    print(
        "[ERRO] Não foi possível carregar o YOLO."
    )

    print(
        erro
    )

    exit()


tempo_modelo = (
    time.perf_counter()
    - inicio_modelo
)


print(
    f"[YOLO] Modelo carregado em "
    f"{tempo_modelo:.2f} segundos."
)

print()


# ============================================================
# CRIAR GERENCIADOR DA CÂMERA
# ============================================================

camera = CameraManager(
    camera_id=CAMERA_ID,
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
    buffer_size=CAMERA_BUFFER_SIZE
)


# ============================================================
# ABRIR CÂMERA
# ============================================================

try:

    camera.open()

except RuntimeError as erro:

    print(
        f"[ERRO] {erro}"
    )

    exit()


# ============================================================
# INICIAR CÂMERA
# ============================================================

camera.start()


print(
    "[CAMERA] Câmera iniciada."
)

print()


# ============================================================
# AGUARDAR PRIMEIRO FRAME
# ============================================================

print(
    "[CAMERA] Aguardando primeiro frame..."
)


while True:

    frame, timestamp = (
        camera.get_latest_frame()
    )

    if frame is not None:

        break

    time.sleep(0.001)


print(
    "[CAMERA] Primeiro frame recebido."
)

print()


# ============================================================
# VARIÁVEIS DO TESTE
# ============================================================

inicio_teste = (
    time.perf_counter()
)

ultimo_timestamp = timestamp

frames_processados = 0

pessoas_detectadas = 0

tempos_inferencia = []

fps_timer = time.perf_counter()

fps_frames = 0

fps_atual = 0


# ============================================================
# LOOP PRINCIPAL
# ============================================================

try:

    while True:

        agora = time.perf_counter()


        # ----------------------------------------------------
        # VERIFICAR TEMPO DO TESTE
        # ----------------------------------------------------

        if (
            agora
            - inicio_teste
            >= DURACAO_TESTE
        ):

            break


        # ----------------------------------------------------
        # PEGAR FRAME MAIS RECENTE
        # ----------------------------------------------------

        frame, timestamp = (
            camera.get_latest_frame()
        )


        if frame is None:

            continue


        # ----------------------------------------------------
        # NÃO PROCESSAR O MESMO FRAME
        # ----------------------------------------------------

        if timestamp == ultimo_timestamp:

            time.sleep(0.001)

            continue


        ultimo_timestamp = timestamp


        # ====================================================
        # INÍCIO DA INFERÊNCIA
        # ====================================================

        inicio_inferencia = (
            time.perf_counter()
        )


        resultados = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            classes=[0],
            verbose=False
        )


        fim_inferencia = (
            time.perf_counter()
        )


        # ====================================================
        # TEMPO DE INFERÊNCIA
        # ====================================================

        tempo_inferencia = (
            fim_inferencia
            -
            inicio_inferencia
        )


        tempos_inferencia.append(
            tempo_inferencia
        )


        frames_processados += 1

        fps_frames += 1


        # ====================================================
        # RESULTADO YOLO
        # ====================================================

        resultado = resultados[0]


        if resultado.boxes is not None:

            quantidade = len(
                resultado.boxes
            )

            pessoas_detectadas += (
                quantidade
            )


        # ====================================================
        # CALCULAR FPS
        # ====================================================

        agora_fps = (
            time.perf_counter()
        )


        if (
            agora_fps
            -
            fps_timer
            >= 1.0
        ):

            fps_atual = fps_frames

            fps_frames = 0

            fps_timer = agora_fps


        # ====================================================
        # DESENHAR RESULTADO
        # ====================================================

        frame_resultado = (
            resultado.plot()
        )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        cv2.putText(
            frame_resultado,
            f"YOLO FPS: {fps_atual}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # TEMPO DE INFERÊNCIA
        # ----------------------------------------------------

        cv2.putText(
            frame_resultado,
            (
                f"Inferencia: "
                f"{tempo_inferencia * 1000:.1f} ms"
            ),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # PESSOAS
        # ----------------------------------------------------

        quantidade_pessoas = 0


        if resultado.boxes is not None:

            quantidade_pessoas = (
                len(resultado.boxes)
            )


        cv2.putText(
            frame_resultado,
            (
                f"Pessoas: "
                f"{quantidade_pessoas}"
            ),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # MOSTRAR
        # ====================================================

        cv2.imshow(
            "VisionAI V2 - YOLO",
            frame_resultado
        )


        # ====================================================
        # Q = SAIR
        # ====================================================

        tecla = (
            cv2.waitKey(1)
            & 0xFF
        )


        if tecla == ord("q"):

            break


finally:

    camera.stop()

    cv2.destroyAllWindows()


# ============================================================
# RESULTADOS
# ============================================================

tempo_total = (
    time.perf_counter()
    -
    inicio_teste
)


print()
print("=" * 60)
print("                 RESULTADO YOLO")
print("=" * 60)
print()


# ============================================================
# FPS MÉDIO
# ============================================================

if tempo_total > 0:

    fps_medio = (
        frames_processados
        /
        tempo_total
    )

else:

    fps_medio = 0.0


print(
    f"Tempo de teste: "
    f"{tempo_total:.2f} segundos"
)

print(
    f"Frames processados: "
    f"{frames_processados}"
)

print(
    f"FPS médio do YOLO: "
    f"{fps_medio:.2f}"
)


# ============================================================
# TEMPO DE INFERÊNCIA
# ============================================================

if len(tempos_inferencia) > 0:

    media = (
        sum(tempos_inferencia)
        /
        len(tempos_inferencia)
    )

    minimo = min(
        tempos_inferencia
    )

    maximo = max(
        tempos_inferencia
    )


    print()

    print(
        f"Inferência média: "
        f"{media * 1000:.2f} ms"
    )

    print(
        f"Inferência mínima: "
        f"{minimo * 1000:.2f} ms"
    )

    print(
        f"Inferência máxima: "
        f"{maximo * 1000:.2f} ms"
    )


# ============================================================
# DETECÇÕES
# ============================================================

print()

print(
    f"Total de detecções de pessoas: "
    f"{pessoas_detectadas}"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("                  TESTE FINALIZADO")
print("=" * 60)
print()