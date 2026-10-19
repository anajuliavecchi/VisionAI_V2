# ============================================================
# VISIONAI V2
# TESTE DE ESTABILIDADE DO INSIGHTFACE
#
# Objetivo:
# Descobrir se existem picos de latência durante o funcionamento
# normal do InsightFace.
#
# Este teste NÃO faz reconhecimento de identidade.
# Ele mede somente a estabilidade da inferência facial.
# ============================================================


# ============================================================
# BIBLIOTECAS
# ============================================================

import time
import statistics
import threading

import cv2
from insightface.app import FaceAnalysis

from config import (
    CAMERA_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_BUFFER_SIZE
)

from camera.camera_manager import CameraManager


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_NAME = "buffalo_sc"

DET_SIZE = (320, 320)

# Duração total do teste
DURACAO_TESTE = 60

# Intervalo entre inferências
AI_INTERVAL = 0.25

# Número de inferências consideradas como warm-up
WARMUP_INFERENCIAS = 5

# Qualquer inferência acima disso será considerada spike
SPIKE_THRESHOLD_MS = 100.0


# ============================================================
# INÍCIO
# ============================================================

print()
print("=" * 70)
print("       VISIONAI V2 | INSIGHTFACE STABILITY TEST")
print("=" * 70)
print()

print(
    "Este teste verifica a estabilidade do InsightFace."
)

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
    f"Duração: {DURACAO_TESTE}s"
)

print(
    f"Warm-up: primeiras "
    f"{WARMUP_INFERENCIAS} inferências"
)

print(
    f"Spike: > {SPIKE_THRESHOLD_MS:.0f} ms"
)

print()


# ============================================================
# CARREGAR INSIGHTFACE
# ============================================================

print(
    "[1/4] Carregando InsightFace..."
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
        "ERRO ao carregar InsightFace:"
    )

    print(
        erro
    )

    exit()


tempo_modelo = (
    time.perf_counter()
    -
    inicio_modelo
)


print(
    f"InsightFace carregado em "
    f"{tempo_modelo:.2f}s."
)

print()


# ============================================================
# CÂMERA
# ============================================================

print(
    "[2/4] Iniciando câmera..."
)

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
        f"ERRO: {erro}"
    )

    exit()


camera.start()


print(
    "Câmera iniciada."
)

print()


# ============================================================
# ESPERAR PRIMEIRO FRAME
# ============================================================

print(
    "Aguardando primeiro frame..."
)


while True:

    frame, timestamp = (
        camera.get_latest_frame()
    )

    if frame is not None:
        break

    time.sleep(0.001)


print(
    "Primeiro frame recebido."
)

print()


# ============================================================
# MÉTRICAS
# ============================================================

warmup_times = []

normal_times = []

all_times = []

spike_times = []

faces_history = []

frame_age_history = []

inference_timestamps = []

# Guarda os números das inferências que apresentaram spike
spike_numbers = []


# ============================================================
# CONTROLE
# ============================================================

running = True

total_inferences = 0

last_inference = 0.0


# ============================================================
# THREAD DA IA
# ============================================================

def ai_thread():

    global running
    global total_inferences
    global last_inference


    print(
        "[3/4] Thread do InsightFace iniciada."
    )

    print()


    while running:

        agora = time.perf_counter()


        # ----------------------------------------------------
        # CONTROLAR FREQUÊNCIA
        # ----------------------------------------------------

        if (
            agora
            -
            last_inference
            <
            AI_INTERVAL
        ):

            time.sleep(0.005)

            continue


        # ----------------------------------------------------
        # FRAME MAIS RECENTE
        # ----------------------------------------------------

        frame, frame_timestamp = (
            camera.get_latest_frame()
        )


        if frame is None:

            continue


        # ----------------------------------------------------
        # IDADE DO FRAME
        # ----------------------------------------------------

        frame_age = (
            time.perf_counter()
            -
            frame_timestamp
        )


        frame_age_history.append(
            frame_age * 1000
        )


        # ----------------------------------------------------
        # REDUZIR IMAGEM
        # ----------------------------------------------------

        small_frame = cv2.resize(
            frame,
            DET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # INFERÊNCIA
        # ----------------------------------------------------

        inicio = time.perf_counter()


        try:

            faces = app.get(
                small_frame
            )

        except Exception as erro:

            print(
                "[ERRO] InsightFace:",
                erro
            )

            continue


        fim = time.perf_counter()


        # ----------------------------------------------------
        # TEMPO
        # ----------------------------------------------------

        inference_ms = (
            fim
            -
            inicio
        ) * 1000


        total_inferences += 1


        numero = total_inferences


        all_times.append(
            inference_ms
        )


        inference_timestamps.append(
            fim
        )


        # ----------------------------------------------------
        # FACES
        # ----------------------------------------------------

        faces_history.append(
            len(faces)
        )


        # ====================================================
        # CLASSIFICAÇÃO
        # ====================================================

        if numero <= WARMUP_INFERENCIAS:

            warmup_times.append(
                inference_ms
            )

            categoria = "WARM-UP"

        else:

            normal_times.append(
                inference_ms
            )

            categoria = "NORMAL"


        # ====================================================
        # SPIKE
        # ====================================================

        if inference_ms > SPIKE_THRESHOLD_MS:

            spike_times.append(
                inference_ms
            )

            spike_numbers.append(
                numero
            )

            print()
            print(
                "⚠️ SPIKE DETECTADO!"
            )

            print(
                f"Inferência #{numero}"
            )

            print(
                f"Tempo: "
                f"{inference_ms:.2f} ms"
            )

            print()


        # ====================================================
        # TERMINAL
        # ====================================================

        print(
            f"#{numero:03d} | "
            f"{categoria:7s} | "
            f"{inference_ms:8.2f} ms | "
            f"Faces: {len(faces)} | "
            f"Frame age: "
            f"{frame_age * 1000:6.2f} ms"
        )


        # ----------------------------------------------------
        # ATUALIZAR TIMER
        # ----------------------------------------------------

        last_inference = (
            time.perf_counter()
        )


    print()
    print(
        "Thread do InsightFace encerrada."
    )


# ============================================================
# INICIAR THREAD
# ============================================================

worker = threading.Thread(
    target=ai_thread,
    daemon=True
)


worker.start()


# ============================================================
# INTERFACE
# ============================================================

print(
    "[4/4] Teste iniciado."
)

print()

print(
    "Faça movimentos naturais diante da câmera."
)

print(
    "Durante o teste:"
)

print(
    "  • fique parado"
)

print(
    "  • vire a cabeça"
)

print(
    "  • mova-se rapidamente"
)

print(
    "  • aproxime-se"
)

print(
    "  • afaste-se"
)

print()

print(
    "O teste termina automaticamente."
)

print()


# ============================================================
# INTERFACE FPS
# ============================================================

interface_frames = 0

interface_fps = 0

fps_timer = time.perf_counter()


# ============================================================
# TESTE
# ============================================================

inicio_teste = time.perf_counter()


try:

    while running:

        agora = time.perf_counter()


        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        if (
            agora
            -
            inicio_teste
            >= DURACAO_TESTE
        ):

            break


        # ----------------------------------------------------
        # FRAME
        # ----------------------------------------------------

        frame, timestamp = (
            camera.get_latest_frame()
        )


        if frame is None:

            continue


        # ----------------------------------------------------
        # FPS DA INTERFACE
        # ----------------------------------------------------

        interface_frames += 1


        if (
            agora
            -
            fps_timer
            >= 1
        ):

            interface_fps = (
                interface_frames
            )

            interface_frames = 0

            fps_timer = agora


        # ----------------------------------------------------
        # MOSTRAR INFORMAÇÕES
        # ----------------------------------------------------

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
            f"Inferencias: {total_inferences}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        if len(all_times) > 0:

            cv2.putText(
                frame,
                (
                    f"IA: "
                    f"{all_times[-1]:.1f} ms"
                ),
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


        cv2.putText(
            frame,
            (
                f"Spikes: "
                f"{len(spike_times)}"
            ),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # MOSTRAR
        # ----------------------------------------------------

        cv2.imshow(
            "VisionAI V2 - InsightFace Stability",
            frame
        )


        # ----------------------------------------------------
        # Q
        # ----------------------------------------------------

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):

            break


finally:

    running = False

    worker.join(
        timeout=5
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
print("          RESULTADO - ESTABILIDADE INSIGHTFACE")
print("=" * 70)
print()


# ============================================================
# GERAIS
# ============================================================

print(
    f"Tempo de teste: "
    f"{tempo_total:.2f}s"
)

print(
    f"Total de inferências: "
    f"{len(all_times)}"
)

print()


# ============================================================
# FUNÇÃO DE ESTATÍSTICAS
# ============================================================

def mostrar_estatisticas(
    titulo,
    valores
):

    print(
        titulo
    )


    if len(valores) == 0:

        print(
            "  Nenhum dado."
        )

        print()

        return


    media = statistics.mean(
        valores
    )

    mediana = statistics.median(
        valores
    )

    minimo = min(
        valores
    )

    maximo = max(
        valores
    )


    valores_ordenados = sorted(
        valores
    )


    # --------------------------------------------------------
    # P95
    # --------------------------------------------------------

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

    print()


    return {
        "media": media,
        "mediana": mediana,
        "p95": p95,
        "minimo": minimo,
        "maximo": maximo
    }


# ============================================================
# WARM-UP
# ============================================================

warmup_stats = mostrar_estatisticas(
    "WARM-UP",
    warmup_times
)


# ============================================================
# OPERAÇÃO NORMAL
# ============================================================

normal_stats = mostrar_estatisticas(
    "OPERAÇÃO NORMAL",
    normal_times
)


# ============================================================
# TODAS AS INFERÊNCIAS
# ============================================================

all_stats = mostrar_estatisticas(
    "TODAS AS INFERÊNCIAS",
    all_times
)


# ============================================================
# SPIKES
# ============================================================

print(
    "SPIKES"
)

print(
    f"  Total: "
    f"{len(spike_times)}"
)


if len(spike_times) > 0:

    print(
        f"  Maior spike: "
        f"{max(spike_times):.2f} ms"
    )

    print(
        "  Inferências:"
    )

    print(
        f"  {spike_numbers}"
    )

else:

    print(
        "  Nenhum spike detectado."
    )


print()


# ============================================================
# FRAME AGE
# ============================================================

if len(frame_age_history) > 0:

    print(
        "IDADE DOS FRAMES"
    )

    print(
        f"  Média: "
        f"{statistics.mean(frame_age_history):.2f} ms"
    )

    print(
        f"  Mediana: "
        f"{statistics.median(frame_age_history):.2f} ms"
    )

    print(
        f"  Máxima: "
        f"{max(frame_age_history):.2f} ms"
    )

    print()


# ============================================================
# DETECÇÃO
# ============================================================

if len(faces_history) > 0:

    print(
        "DETECÇÃO"
    )

    print(
        f"  Média de faces: "
        f"{statistics.mean(faces_history):.2f}"
    )

    print(
        f"  Máximo de faces: "
        f"{max(faces_history)}"
    )

    print()


# ============================================================
# CÂMERA
# ============================================================

camera_metrics = (
    camera.get_metrics()
)


frames_captured = (
    camera_metrics[
        "frames_captured"
    ]
)


if tempo_total > 0:

    camera_fps = (
        frames_captured
        /
        tempo_total
    )

else:

    camera_fps = 0


print(
    "CÂMERA"
)

print(
    f"  Frames capturados: "
    f"{frames_captured}"
)

print(
    f"  FPS médio: "
    f"{camera_fps:.2f}"
)

print()


# ============================================================
# DIAGNÓSTICO
# ============================================================

print("=" * 70)
print("                    DIAGNÓSTICO")
print("=" * 70)
print()


if normal_stats:

    normal_p95 = (
        normal_stats["p95"]
    )

    normal_max = (
        normal_stats["maximo"]
    )


    # --------------------------------------------------------
    # P95
    # --------------------------------------------------------

    if normal_p95 < 50:

        print(
            "🟢 Operação normal: "
            "P95 excelente."
        )

    elif normal_p95 < 100:

        print(
            "🟡 Operação normal: "
            "P95 moderado."
        )

    else:

        print(
            "🔴 Operação normal: "
            "P95 elevado."
        )


    # --------------------------------------------------------
    # SPIKES
    # --------------------------------------------------------

    if len(spike_times) == 0:

        print(
            "🟢 Nenhum spike detectado."
        )

    else:

        print(
            f"🟡 {len(spike_times)} spike(s) "
            f"detectado(s)."
        )


    # --------------------------------------------------------
    # DIFERENÇA WARM-UP/NORMAL
    # --------------------------------------------------------

    if (
        warmup_stats
        and
        normal_stats
    ):

        warmup_media = (
            warmup_stats["media"]
        )

        normal_media = (
            normal_stats["media"]
        )


        print()

        print(
            "COMPARAÇÃO WARM-UP × NORMAL"
        )

        print(
            f"  Warm-up médio: "
            f"{warmup_media:.2f} ms"
        )

        print(
            f"  Normal médio: "
            f"{normal_media:.2f} ms"
        )


        if warmup_media > (
            normal_media * 2
        ):

            print(
                "  🟡 Warm-up "
                "significativamente mais lento."
            )

        else:

            print(
                "  🟢 Warm-up "
                "não apresentou diferença extrema."
            )


print()

print(
    "IMPORTANTE:"
)

print(
    "Este teste mede estabilidade e latência "
    "da detecção facial."
)

print(
    "Ele NÃO mede precisão de identidade."
)

print()

print("=" * 70)
print("                    FIM DO TESTE")
print("=" * 70)
print()