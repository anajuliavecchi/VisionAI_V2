# ============================================================
# VISIONAI V2
# TESTE DE RECONHECIMENTO FACIAL
#
# Objetivo:
# Medir o reconhecimento facial puro.
#
# Este teste NÃO possui:
# - tracking
# - previsão
# - suavização
# - confirmação temporal
#
# Queremos medir o motor de reconhecimento isoladamente.
# ============================================================


# ============================================================
# BIBLIOTECAS
# ============================================================

import time
import statistics

import cv2

from insightface.app import FaceAnalysis

from config import (
    CAMERA_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_BUFFER_SIZE
)

from camera.camera_manager import CameraManager

from recognition.face_database import FaceDatabase
from recognition.face_recognizer import FaceRecognizer


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_NAME = "buffalo_sc"

DET_SIZE = (320, 320)

AI_INTERVAL = 0.25

THRESHOLD = 0.40

DURACAO_TESTE = 60


# ============================================================
# INÍCIO
# ============================================================

print()
print("=" * 70)
print("          VISIONAI V2 | TESTE DE RECONHECIMENTO")
print("=" * 70)
print()


# ============================================================
# BANCO
# ============================================================

print(
    "[1/5] Carregando banco de faces..."
)

database_manager = FaceDatabase()

database = database_manager.load()


print()

print(
    f"Pessoas cadastradas: "
    f"{len(database_manager.get_people())}"
)

print(
    f"Embeddings carregados: "
    f"{database_manager.get_total_embeddings()}"
)

print()

for pessoa in database_manager.get_people():

    print(
        f"  - {pessoa}: "
        f"{len(database[pessoa])} embeddings"
    )

print()


# ============================================================
# RECONHECEDOR
# ============================================================

recognizer = FaceRecognizer(
    database=database,
    threshold=THRESHOLD
)


# ============================================================
# INSIGHTFACE
# ============================================================

print(
    "[2/5] Carregando InsightFace..."
)

inicio_modelo = time.perf_counter()


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
    "[3/5] Iniciando câmera..."
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

    exit(1)


camera.start()


print(
    "Câmera iniciada."
)

print()


# ============================================================
# MÉTRICAS
# ============================================================

inference_times = []

recognition_times = []

scores = []

margins = []

names_history = []

faces_history = []

frame_ages = []


# ============================================================
# CONTROLE
# ============================================================

last_inference = 0

total_inferences = 0

running = True


# ============================================================
# TESTE
# ============================================================

print(
    "[4/5] Teste iniciado."
)

print()

print(
    "Faça movimentos naturais diante da câmera."
)

print(
    "Teste primeiro parado."
)

print(
    "Depois vire a cabeça."
)

print(
    "Depois movimente-se."
)

print()

print(
    "Pressione Q para encerrar antes dos 60 segundos."
)

print()


inicio_teste = time.perf_counter()


try:

    while running:

        agora = time.perf_counter()


        # ----------------------------------------------------
        # TEMPO
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
        # IDADE DO FRAME
        # ----------------------------------------------------

        frame_age = (
            time.perf_counter()
            -
            timestamp
        ) * 1000


        frame_ages.append(
            frame_age
        )


        # ====================================================
        # INFERÊNCIA
        # ====================================================

        if (
            agora
            -
            last_inference
            >=
            AI_INTERVAL
        ):

            small_frame = cv2.resize(
                frame,
                DET_SIZE,
                interpolation=cv2.INTER_LINEAR
            )


            # ------------------------------------------------
            # INSIGHTFACE
            # ------------------------------------------------

            inicio_ia = (
                time.perf_counter()
            )


            faces = app.get(
                small_frame
            )


            fim_ia = (
                time.perf_counter()
            )


            inference_ms = (
                fim_ia
                -
                inicio_ia
            ) * 1000


            inference_times.append(
                inference_ms
            )


            faces_history.append(
                len(faces)
            )


            total_inferences += 1


            # =================================================
            # RECONHECER CADA ROSTO
            # =================================================

            for face in faces:

                embedding = (
                    face.embedding
                )


                if embedding is None:

                    continue


                inicio_reconhecimento = (
                    time.perf_counter()
                )


                resultado = (
                    recognizer.identify(
                        embedding
                    )
                )


                fim_reconhecimento = (
                    time.perf_counter()
                )


                recognition_ms = (
                    fim_reconhecimento
                    -
                    inicio_reconhecimento
                ) * 1000


                recognition_times.append(
                    recognition_ms
                )


                nome = resultado[
                    "name"
                ]

                score = resultado[
                    "score"
                ]

                second_score = resultado[
                    "second_score"
                ]

                margin = resultado[
                    "margin"
                ]


                scores.append(
                    score
                )

                margins.append(
                    margin
                )

                names_history.append(
                    nome
                )


                print(
                    f"[{total_inferences:03d}] "
                    f"{nome.upper():12s} | "
                    f"Score: {score:.4f} | "
                    f"2º: {second_score:.4f} | "
                    f"Margem: {margin:.4f} | "
                    f"IA: {inference_ms:.2f} ms | "
                    f"Recon: {recognition_ms:.3f} ms"
                )


            last_inference = (
                time.perf_counter()
            )


        # ====================================================
        # INTERFACE
        # ====================================================

        cv2.putText(
            frame,
            (
                f"Inferencias: "
                f"{total_inferences}"
            ),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        if len(names_history) > 0:

            ultimo_nome = (
                names_history[-1]
            )

            ultimo_score = (
                scores[-1]
            )


            cv2.putText(
                frame,
                (
                    f"{ultimo_nome.upper()} "
                    f"{ultimo_score:.3f}"
                ),
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        cv2.imshow(
            "VisionAI V2 - Recognition Test",
            frame
        )


        # ====================================================
        # SAÍDA
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):

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
print("=" * 70)
print("             RESULTADO DO RECONHECIMENTO")
print("=" * 70)
print()


print(
    f"Tempo de teste: "
    f"{tempo_total:.2f}s"
)

print(
    f"Total de inferências: "
    f"{total_inferences}"
)

print()


# ============================================================
# FUNÇÃO ESTATÍSTICA
# ============================================================

def estatisticas(
    valores
):

    if len(valores) == 0:

        return None


    ordenados = sorted(
        valores
    )


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


    indice_p95 = int(
        0.95
        *
        (
            len(ordenados)
            -
            1
        )
    )


    p95 = ordenados[
        indice_p95
    ]


    return {
        "media": media,
        "mediana": mediana,
        "p95": p95,
        "minimo": minimo,
        "maximo": maximo
    }


# ============================================================
# INFERENCE
# ============================================================

stats_ia = estatisticas(
    inference_times
)


if stats_ia:

    print(
        "TEMPO DO INSIGHTFACE"
    )

    print(
        f"  Média:   "
        f"{stats_ia['media']:.2f} ms"
    )

    print(
        f"  Mediana: "
        f"{stats_ia['mediana']:.2f} ms"
    )

    print(
        f"  P95:     "
        f"{stats_ia['p95']:.2f} ms"
    )

    print(
        f"  Mínimo:  "
        f"{stats_ia['minimo']:.2f} ms"
    )

    print(
        f"  Máximo:  "
        f"{stats_ia['maximo']:.2f} ms"
    )

    print()


# ============================================================
# RECONHECIMENTO
# ============================================================

stats_rec = estatisticas(
    recognition_times
)


if stats_rec:

    print(
        "TEMPO DO RECONHECIMENTO"
    )

    print(
        f"  Média:   "
        f"{stats_rec['media']:.4f} ms"
    )

    print(
        f"  Mediana: "
        f"{stats_rec['mediana']:.4f} ms"
    )

    print(
        f"  P95:     "
        f"{stats_rec['p95']:.4f} ms"
    )

    print(
        f"  Mínimo:  "
        f"{stats_rec['minimo']:.4f} ms"
    )

    print(
        f"  Máximo:  "
        f"{stats_rec['maximo']:.4f} ms"
    )

    print()


# ============================================================
# SCORE
# ============================================================

if len(scores) > 0:

    score_stats = estatisticas(
        scores
    )


    print(
        "SCORES"
    )

    print(
        f"  Média:   "
        f"{score_stats['media']:.4f}"
    )

    print(
        f"  Mediana: "
        f"{score_stats['mediana']:.4f}"
    )

    print(
        f"  P95:     "
        f"{score_stats['p95']:.4f}"
    )

    print(
        f"  Mínimo:  "
        f"{score_stats['minimo']:.4f}"
    )

    print(
        f"  Máximo:  "
        f"{score_stats['maximo']:.4f}"
    )

    print()


# ============================================================
# MARGEM
# ============================================================

if len(margins) > 0:

    margin_stats = estatisticas(
        margins
    )


    print(
        "MARGEM ENTRE 1º E 2º CANDIDATO"
    )

    print(
        f"  Média:   "
        f"{margin_stats['media']:.4f}"
    )

    print(
        f"  Mediana: "
        f"{margin_stats['mediana']:.4f}"
    )

    print(
        f"  P95:     "
        f"{margin_stats['p95']:.4f}"
    )

    print(
        f"  Mínimo:  "
        f"{margin_stats['minimo']:.4f}"
    )

    print()


# ============================================================
# DISTRIBUIÇÃO DAS IDENTIDADES
# ============================================================

print(
    "RESULTADOS POR IDENTIDADE"
)

contagem = {}


for nome in names_history:

    contagem[nome] = (
        contagem.get(nome, 0)
        +
        1
    )


for nome, quantidade in sorted(
    contagem.items()
):

    percentual = (
        quantidade
        /
        len(names_history)
        *
        100
    )


    print(
        f"  {nome:15s} "
        f"{quantidade:4d} "
        f"({percentual:5.1f}%)"
    )


print()


# ============================================================
# IDADE DOS FRAMES
# ============================================================

if len(frame_ages) > 0:

    frame_stats = estatisticas(
        frame_ages
    )


    print(
        "IDADE DOS FRAMES"
    )

    print(
        f"  Média:   "
        f"{frame_stats['media']:.2f} ms"
    )

    print(
        f"  Mediana: "
        f"{frame_stats['mediana']:.2f} ms"
    )

    print(
        f"  Máxima:  "
        f"{frame_stats['maximo']:.2f} ms"
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
# DIAGNÓSTICO
# ============================================================

print("=" * 70)
print("                    DIAGNÓSTICO")
print("=" * 70)
print()


if stats_ia:

    if stats_ia["p95"] < 50:

        print(
            "🟢 InsightFace: latência excelente."
        )

    elif stats_ia["p95"] < 100:

        print(
            "🟡 InsightFace: latência moderada."
        )

    else:

        print(
            "🔴 InsightFace: latência elevada."
        )


if stats_rec:

    if stats_rec["p95"] < 1:

        print(
            "🟢 Comparação dos embeddings: "
            "extremamente rápida."
        )

    elif stats_rec["p95"] < 5:

        print(
            "🟢 Comparação dos embeddings: "
            "rápida."
        )

    else:

        print(
            "🟡 Comparação dos embeddings: "
            "merece investigação."
        )


if len(frame_ages) > 0:

    if statistics.mean(frame_ages) < 50:

        print(
            "🟢 Frames usados pela IA "
            "estão atualizados."
        )

    else:

        print(
            "🔴 Frames usados pela IA "
            "estão chegando atrasados."
        )


print()

print(
    "IMPORTANTE:"
)

print(
    "Este teste mede o comportamento "
    "do reconhecimento."
)

print(
    "Ele ainda NÃO mede precisão real "
    "contra uma verdade-terreno."
)

print(
    "Para isso faremos um benchmark "
    "controlado posteriormente."
)

print()

print("=" * 70)
print("                       FIM")
print("=" * 70)