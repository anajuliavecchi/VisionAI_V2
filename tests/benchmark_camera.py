# ============================================================
# VISIONAI V2
# BENCHMARK REAL DA CÂMERA
# ============================================================

import time

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
# CONFIGURAÇÕES DO TESTE
# ============================================================

DURACAO_TESTE = 20


# ============================================================
# INÍCIO
# ============================================================

print()
print("=" * 60)
print("       VISIONAI V2 | BENCHMARK REAL DA CÂMERA")
print("=" * 60)
print()

print(
    f"Duração: {DURACAO_TESTE} segundos"
)

print(
    f"Resolução desejada: "
    f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
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
    "[TESTE] Câmera iniciada."
)

print(
    "[TESTE] Capturando frames..."
)

print()


# ============================================================
# AGUARDAR PRIMEIRO FRAME
# ============================================================

print(
    "[TESTE] Aguardando primeiro frame..."
)


while True:

    frame, timestamp = (
        camera.get_latest_frame()
    )


    if frame is not None:

        break


    time.sleep(0.001)


print(
    "[TESTE] Primeiro frame recebido."
)

print()


# ============================================================
# CONTADOR INICIAL
# ============================================================

frames_iniciais = (
    camera.get_metrics()
    ["frames_captured"]
)


# ============================================================
# INÍCIO DO TESTE
# ============================================================

inicio = time.perf_counter()


# ============================================================
# LOOP DO BENCHMARK
# ============================================================

while True:

    agora = time.perf_counter()


    tempo_decorrido = (
        agora - inicio
    )


    # --------------------------------------------------------
    # ENCERRAR APÓS O TEMPO DEFINIDO
    # --------------------------------------------------------

    if tempo_decorrido >= DURACAO_TESTE:

        break


    # --------------------------------------------------------
    # A THREAD DA CÂMERA CONTINUA
    # CAPTURANDO INDEPENDENTEMENTE
    # --------------------------------------------------------

    time.sleep(0.01)


# ============================================================
# TEMPO FINAL
# ============================================================

fim = time.perf_counter()


tempo_total = (
    fim - inicio
)


# ============================================================
# CONTADOR FINAL
# ============================================================

frames_finais = (
    camera.get_metrics()
    ["frames_captured"]
)


# ============================================================
# FINALIZAR
# ============================================================

camera.stop()


# ============================================================
# CALCULAR FRAMES
# ============================================================

frames_capturados = (
    frames_finais
    -
    frames_iniciais
)


# ============================================================
# CALCULAR FPS
# ============================================================

if tempo_total > 0:

    fps_captura = (
        frames_capturados
        /
        tempo_total
    )

else:

    fps_captura = 0.0


# ============================================================
# RESULTADOS
# ============================================================

print()
print("=" * 60)
print("              RESULTADO DO BENCHMARK")
print("=" * 60)
print()

print(
    f"Tempo de teste: "
    f"{tempo_total:.2f} segundos"
)

print(
    f"Frames realmente capturados: "
    f"{frames_capturados}"
)

print(
    f"FPS real da captura: "
    f"{fps_captura:.2f}"
)

print()

print(
    "Este FPS representa a quantidade de"
)

print(
    "frames realmente entregues pela câmera."
)

print()

print("=" * 60)
print("                 TESTE FINALIZADO")
print("=" * 60)
print()