# ============================================================
# VISIONAI V2
# TESTE PROFISSIONAL - INSIGHTFACE
#
# Objetivo:
# Medir o comportamento do InsightFace sozinho,
# utilizando captura assíncrona e somente o frame mais recente.
#
# NÃO existe reconhecimento facial neste teste.
# Estamos medindo DETECÇÃO FACIAL + CUSTO COMPUTACIONAL.
# ============================================================


# ============================================================
# BIBLIOTECAS
# ============================================================

import time
import threading
import statistics

import cv2
import numpy as np

from insightface.app import FaceAnalysis

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
# CPU
# ============================================================

try:

    import psutil

    PSUTIL_AVAILABLE = True

except ImportError:

    PSUTIL_AVAILABLE = False


# ============================================================
# CONFIGURAÇÕES DO TESTE
# ============================================================

MODEL_NAME = "buffalo_sc"

DET_SIZE = (320, 320)

DURACAO_TESTE = 30

# ============================================================
# FREQUÊNCIA DA IA
# ============================================================
#
# 0.25 segundo = aproximadamente 4 inferências por segundo.
#
# Isso é proposital.
#
# Não queremos obrigar o InsightFace a acompanhar os ~30 FPS
# da câmera.
#
# Queremos descobrir quanto custa executar o reconhecimento
# facial de forma controlada.
# ============================================================

AI_INTERVAL = 0.25


# ============================================================
# LIMITE PARA CONSIDERAR UM PICO
# ============================================================

# Inferências acima disso serão contabilizadas como pico.

SPIKE_THRESHOLD_MS = 100.0


# ============================================================
# INÍCIO
# ============================================================

print()
print("=" * 70)
print("          VISIONAI V2 | TESTE INSIGHTFACE")
print("=" * 70)
print()

print(
    "Objetivo:"
)

print(
    "Medir o custo real do InsightFace "
    "sem reconhecimento facial."
)

print()

print(
    f"Modelo: {MODEL_NAME}"
)

print(
    f"Detector: {DET_SIZE[0]}x{DET_SIZE[1]}"
)

print(
    f"Intervalo da IA: {AI_INTERVAL:.2f}s"
)

print(
    f"Frequência teórica: "
    f"{1 / AI_INTERVAL:.1f} inferências/s"
)

print(
    f"Duração: {DURACAO_TESTE}s"
)

print()


if PSUTIL_AVAILABLE:

    print(
        "[CPU] psutil disponível."
    )

else:

    print(
        "[CPU] psutil não está instalado."
    )

    print(
        "[CPU] O teste continuará, "
        "mas sem percentual de CPU."
    )

print()


# ============================================================
# CARREGAR INSIGHTFACE
# ============================================================

print(
    "[INSIGHTFACE] Carregando modelo..."
)

inicio_modelo = time.perf_counter()


try:

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


except Exception as erro:

    print()
    print(
        "[ERRO] Não foi possível carregar "
        "o InsightFace."
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
    f"[INSIGHTFACE] Modelo carregado em "
    f"{tempo_modelo:.2f}s."
)

print()


# ============================================================
# CÂMERA
# ============================================================

camera = CameraManager(
    camera_id=CAMERA_ID,
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
    buffer_size=CAMERA_BUFFER_SIZE
)


try:

    camera.open()

except RuntimeError as erro:

    print(
        f"[ERRO] {erro}"
    )

    exit()


camera.start()


print(
    "[CAMERA] Thread de captura iniciada."
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

running = True

latest_result = []

result_lock = threading.Lock()


# ============================================================
# MÉTRICAS
# ============================================================

inference_times = []

inference_timestamps = []

frame_ages = []

faces_per_inference = []

cpu_samples = []

spikes = 0

total_inferences = 0

total_faces = 0

frames_skipped_by_scheduler = 0


# ============================================================
# CONTROLE DE TEMPO
# ============================================================

teste_inicio = time.perf_counter()

last_inference = 0.0


# ============================================================
# THREAD DA IA
# ============================================================

def ai_worker():

    global running
    global latest_result
    global last_inference

    global spikes
    global total_inferences
    global total_faces

    print(
        "[AI] Thread do InsightFace iniciada."
    )

    print()


    while running:

        agora = time.perf_counter()


        # ====================================================
        # CONTROLAR FREQUÊNCIA
        # ====================================================

        if (
            agora
            -
            last_inference
            <
            AI_INTERVAL
        ):

            time.sleep(0.005)

            continue


        # ====================================================
        # PEGAR FRAME MAIS RECENTE
        # ====================================================

        frame, frame_timestamp = (
            camera.get_latest_frame()
        )


        if frame is None:

            continue


        # ====================================================
        # TEMPO DO FRAME
        # ====================================================

        agora_antes_ia = (
            time.perf_counter()
        )


        frame_age = (
            agora_antes_ia
            -
            frame_timestamp
        )


        frame_ages.append(
            frame_age
        )


        # ====================================================
        # REDUZIR IMAGEM
        # ====================================================

        small_frame = cv2.resize(
            frame,
            DET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )


        # ====================================================
        # INFERÊNCIA
        # ====================================================

        inicio = time.perf_counter()


        try:

            faces = app.get(
                small_frame
            )

        except Exception as erro:

            print(
                "[AI] Erro durante inferência:",
                erro
            )

            continue


        fim = time.perf_counter()


        # ====================================================
        # TEMPO
        # ====================================================

        inference_time = (
            fim
            -
            inicio
        )


        inference_ms = (
            inference_time
            *
            1000
        )


        inference_times.append(
            inference_ms
        )


        inference_timestamps.append(
            fim
        )


        total_inferences += 1


        # ====================================================
        # ROSTOS
        # ====================================================

        quantidade_faces = len(
            faces
        )


        faces_per_inference.append(
            quantidade_faces
        )


        total_faces += (
            quantidade_faces
        )


        # ====================================================
        # DETECTAR PICO
        # ====================================================

        if inference_ms > SPIKE_THRESHOLD_MS:

            spikes += 1


        # ====================================================
        # CPU
        # ====================================================

        if PSUTIL_AVAILABLE:

            cpu = psutil.cpu_percent(
                interval=None
            )

            cpu_samples.append(
                cpu
            )


        # ====================================================
        # GUARDAR RESULTADO
        # ====================================================

        with result_lock:

            latest_result = faces


        # ====================================================
        # ATUALIZAR TIMER
        # ====================================================

        last_inference = (
            time.perf_counter()
        )


        # ====================================================
        # TERMINAL
        # ====================================================

        print(
            f"[AI] "
            f"Inferência: "
            f"{inference_ms:7.2f} ms | "
            f"Faces: "
            f"{quantidade_faces} | "
            f"Frame age: "
            f"{frame_age * 1000:7.2f} ms"
        )


    print(
        "[AI] Thread do InsightFace encerrada."
    )


# ============================================================
# INICIAR THREAD
# ============================================================

ai_thread = threading.Thread(
    target=ai_worker,
    daemon=True
)


ai_thread.start()


# ============================================================
# INTERFACE
# ============================================================

print(
    "[SYSTEM] VisionAI V2 iniciado."
)

print(
    "[SYSTEM] O teste vai durar "
    f"{DURACAO_TESTE} segundos."
)

print()

print(
    "Durante o teste:"
)

print(
    "1. Fique parado."
)

print(
    "2. Vire o rosto lentamente."
)

print(
    "3. Mova a cabeça rapidamente."
)

print(
    "4. Aproxime-se da câmera."
)

print(
    "5. Afaste-se da câmera."
)

print()

print(
    "Pressione Q se quiser encerrar antes."
)

print()


# ============================================================
# FPS DA INTERFACE
# ============================================================

interface_frames = 0

interface_timer = (
    time.perf_counter()
)

interface_fps = 0


# ============================================================
# INÍCIO DO TESTE
# ============================================================

inicio_teste = (
    time.perf_counter()
)


# ============================================================
# LOOP PRINCIPAL
# ============================================================

try:

    while running:

        agora = (
            time.perf_counter()
        )


        # ----------------------------------------------------
        # TEMPO DO TESTE
        # ----------------------------------------------------

        if (
            agora
            -
            inicio_teste
            >= DURACAO_TESTE
        ):

            break


        # ====================================================
        # FRAME MAIS RECENTE
        # ====================================================

        frame, frame_timestamp = (
            camera.get_latest_frame()
        )


        if frame is None:

            continue


        # ====================================================
        # RESULTADO MAIS RECENTE
        # ====================================================

        with result_lock:

            faces = list(
                latest_result
            )


        # ====================================================
        # ESCALA
        # ====================================================

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


        # ====================================================
        # DESENHAR ROSTOS
        # ====================================================

        for face in faces:

            bbox = (
                face.bbox.astype(int)
            )


            x1, y1, x2, y2 = bbox


            x1 = int(
                x1 * scale_x
            )

            x2 = int(
                x2 * scale_x
            )

            y1 = int(
                y1 * scale_y
            )

            y2 = int(
                y2 * scale_y
            )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            score = float(
                face.det_score
            )


            cv2.putText(
                frame,
                f"Face {score:.2f}",
                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # FPS DA INTERFACE
        # ====================================================

        interface_frames += 1


        agora_fps = (
            time.perf_counter()
        )


        if (
            agora_fps
            -
            interface_timer
            >= 1.0
        ):

            interface_fps = (
                interface_frames
            )

            interface_frames = 0

            interface_timer = (
                agora_fps
            )


        # ====================================================
        # INFORMAÇÕES NA TELA
        # ====================================================

        cv2.putText(
            frame,
            f"Camera FPS: {interface_fps}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        if len(inference_times) > 0:

            ultima_inferencia = (
                inference_times[-1]
            )

            cv2.putText(
                frame,
                (
                    f"IA: "
                    f"{ultima_inferencia:.1f} ms"
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
            "VisionAI V2 - InsightFace",
            frame
        )


        # ====================================================
        # Q = SAIR
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):

            break


finally:

    running = False

    ai_thread.join(
        timeout=3.0
    )

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
print("=" * 70)
print("             RESULTADO INSIGHTFACE")
print("=" * 70)
print()


# ============================================================
# INFORMAÇÕES GERAIS
# ============================================================

print(
    f"Tempo de teste: "
    f"{tempo_total:.2f} segundos"
)

print(
    f"Total de inferências: "
    f"{total_inferences}"
)

print(
    f"Total de faces detectadas: "
    f"{total_faces}"
)

print()


# ============================================================
# FPS DA IA
# ============================================================

if tempo_total > 0:

    ai_fps = (
        total_inferences
        /
        tempo_total
    )

else:

    ai_fps = 0


print(
    f"FPS da IA: "
    f"{ai_fps:.2f}"
)


# ============================================================
# TEMPOS DE INFERÊNCIA
# ============================================================

if len(inference_times) > 0:

    media = statistics.mean(
        inference_times
    )

    minimo = min(
        inference_times
    )

    maximo = max(
        inference_times
    )

    mediana = statistics.median(
        inference_times
    )


    # --------------------------------------------------------
    # P95
    # --------------------------------------------------------

    valores_ordenados = sorted(
        inference_times
    )


    indice_p95 = int(
        0.95
        *
        (
            len(valores_ordenados)
            -
            1
        )
    )


    p95 = valores_ordenados[
        indice_p95
    ]


    print()

    print(
        "TEMPO DE INFERÊNCIA"
    )

    print(
        f"  Média:   {media:.2f} ms"
    )

    print(
        f"  Mediana: {mediana:.2f} ms"
    )

    print(
        f"  P95:     {p95:.2f} ms"
    )

    print(
        f"  Mínimo:  {minimo:.2f} ms"
    )

    print(
        f"  Máximo:  {maximo:.2f} ms"
    )


# ============================================================
# PICOS
# ============================================================

print()

print(
    f"Picos > "
    f"{SPIKE_THRESHOLD_MS:.0f} ms: "
    f"{spikes}"
)


# ============================================================
# IDADE DOS FRAMES
# ============================================================

if len(frame_ages) > 0:

    idade_media = (
        statistics.mean(
            frame_ages
        )
        *
        1000
    )

    idade_maxima = (
        max(
            frame_ages
        )
        *
        1000
    )


    idade_mediana = (
        statistics.median(
            frame_ages
        )
        *
        1000
    )


    print()

    print(
        "IDADE DO FRAME USADO PELA IA"
    )

    print(
        f"  Média:   "
        f"{idade_media:.2f} ms"
    )

    print(
        f"  Mediana: "
        f"{idade_mediana:.2f} ms"
    )

    print(
        f"  Máxima:  "
        f"{idade_maxima:.2f} ms"
    )


# ============================================================
# FACES
# ============================================================

if len(faces_per_inference) > 0:

    media_faces = (
        statistics.mean(
            faces_per_inference
        )
    )


    print()

    print(
        "DETECÇÃO"
    )

    print(
        f"  Média de faces por inferência: "
        f"{media_faces:.2f}"
    )

    print(
        f"  Maior quantidade em uma inferência: "
        f"{max(faces_per_inference)}"
    )


# ============================================================
# CPU
# ============================================================

if PSUTIL_AVAILABLE and len(cpu_samples) > 0:

    cpu_media = (
        statistics.mean(
            cpu_samples
        )
    )

    cpu_maxima = max(
        cpu_samples
    )


    print()

    print(
        "CPU"
    )

    print(
        f"  Uso médio observado: "
        f"{cpu_media:.1f}%"
    )

    print(
        f"  Maior uso observado: "
        f"{cpu_maxima:.1f}%"
    )

else:

    print()

    print(
        "CPU"
    )

    print(
        "  Métrica não disponível."
    )


# ============================================================
# CÂMERA
# ============================================================

metricas_camera = (
    camera.get_metrics()
)


frames_camera = (
    metricas_camera[
        "frames_captured"
    ]
)


if tempo_total > 0:

    fps_camera = (
        frames_camera
        /
        tempo_total
    )

else:

    fps_camera = 0


print()

print(
    "CÂMERA"
)

print(
    f"  Frames capturados: "
    f"{frames_camera}"
)

print(
    f"  FPS da captura: "
    f"{fps_camera:.2f}"
)


# ============================================================
# CONCLUSÃO AUTOMÁTICA
# ============================================================

print()
print("=" * 70)
print("                 DIAGNÓSTICO")
print("=" * 70)
print()


# ------------------------------------------------------------
# Inferência
# ------------------------------------------------------------

if len(inference_times) > 0:

    if media < 50:

        print(
            "🟢 Inferência média: excelente."
        )

    elif media < 100:

        print(
            "🟡 Inferência média: moderada."
        )

    else:

        print(
            "🔴 Inferência média: alta."
        )


    # --------------------------------------------------------
    # P95
    # --------------------------------------------------------

    if p95 < 100:

        print(
            "🟢 P95: comportamento estável."
        )

    elif p95 < 200:

        print(
            "🟡 P95: existem algumas "
            "inferências lentas."
        )

    else:

        print(
            "🔴 P95: existem picos "
            "importantes de latência."
        )


# ------------------------------------------------------------
# Picos
# ------------------------------------------------------------

if spikes == 0:

    print(
        "🟢 Nenhum pico acima do limite."
    )

else:

    print(
        f"🟡 Foram registrados "
        f"{spikes} picos acima de "
        f"{SPIKE_THRESHOLD_MS:.0f} ms."
    )


# ------------------------------------------------------------
# CPU
# ------------------------------------------------------------

if PSUTIL_AVAILABLE and len(cpu_samples) > 0:

    if cpu_media < 60:

        print(
            "🟢 CPU: carga média confortável."
        )

    elif cpu_media < 85:

        print(
            "🟡 CPU: carga média elevada."
        )

    else:

        print(
            "🔴 CPU: carga média muito elevada."
        )


print()
print(
    "Este diagnóstico NÃO avalia a precisão "
    "do reconhecimento facial."
)

print(
    "Ele avalia somente o custo e a estabilidade "
    "da detecção facial."
)

print()
print("=" * 70)
print("                 TESTE FINALIZADO")
print("=" * 70)
print()