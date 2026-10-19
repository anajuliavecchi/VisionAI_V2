import sys
import os
import cv2
import time


# ============================================================
# LOCALIZAR RAIZ DO PROJETO
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
    FACES_DIR,
    CAMERA_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    DET_SIZE,
    RECOGNITION_THRESHOLD
)


from face.face_database import (
    FaceDatabase
)


from face.recognition_engine import (
    RecognitionEngine
)


# ============================================================
# INÍCIO
# ============================================================

print()
print("=" * 70)
print(
    "       VISIONAI V2 - RECONHECIMENTO BASELINE"
)
print("=" * 70)
print()


# ============================================================
# CARREGAR BANCO
# ============================================================

print(
    "Carregando banco de faces..."
)

print()


database = FaceDatabase(
    FACES_DIR
)


database.load()


print(
    f"Pessoas cadastradas: "
    f"{len(database.people())}"
)


print(
    f"Embeddings carregados: "
    f"{database.total_embeddings()}"
)


print()


# ============================================================
# MOSTRAR PESSOAS
# ============================================================

for pessoa in database.people():

    quantidade = len(
        database.get_embeddings(
            pessoa
        )
    )


    print(
        f"  {pessoa.upper():<12} "
        f"{quantidade} embeddings"
    )


# ============================================================
# VERIFICAR BANCO
# ============================================================

if len(
    database.people()
) == 0:

    print()
    print(
        "ERRO: nenhuma pessoa cadastrada."
    )

    raise SystemExit


print()


# ============================================================
# MOTOR
# ============================================================

engine = RecognitionEngine(
    database.database,
    threshold=RECOGNITION_THRESHOLD
)


# ============================================================
# INSIGHTFACE
# ============================================================

print(
    "Carregando InsightFace..."
)


start_model = time.perf_counter()


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


model_time = (
    time.perf_counter()
    -
    start_model
)


print(
    f"InsightFace carregado em "
    f"{model_time:.2f} segundos."
)


print()


# ============================================================
# CÂMERA
# ============================================================

print(
    f"Abrindo câmera ID {CAMERA_ID}..."
)


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

    print()
    print(
        "ERRO: não foi possível abrir a câmera."
    )

    raise SystemExit


print(
    "Câmera iniciada."
)


print()
print(
    "================================================"
)

print(
    "CONTROLES"
)

print(
    "Q = sair"
)

print(
    "================================================"
)

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

        print(
            "ERRO: falha ao capturar frame."
        )

        continue


    # ========================================================
    # REDUZIR IMAGEM
    # ========================================================

    small_frame = cv2.resize(
        frame,
        DET_SIZE,
        interpolation=cv2.INTER_LINEAR
    )


    # ========================================================
    # INSIGHTFACE
    # ========================================================

    start_inference = (
        time.perf_counter()
    )


    try:

        faces = app.get(
            small_frame
        )


    except Exception as erro:

        print(
            "Erro no InsightFace:",
            erro
        )

        continue


    inference_ms = (
        time.perf_counter()
        -
        start_inference
    ) * 1000


    # ========================================================
    # ESCALA
    # ========================================================

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
    # PROCESSAR TODAS AS FACES
    # ========================================================

    for index, face in enumerate(
        faces
    ):

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        embedding = (
            face.embedding
        )


        if embedding is None:

            continue


        # ----------------------------------------------------
        # RECONHECIMENTO
        # ----------------------------------------------------

        result = (
            engine.recognize(
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


        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        x1, y1, x2, y2 = (
            face.bbox.astype(float)
        )


        x1 *= scale_x
        x2 *= scale_x

        y1 *= scale_y
        y2 *= scale_y


        x1 = int(x1)
        y1 = int(y1)

        x2 = int(x2)
        y2 = int(y2)


        # ====================================================
        # STATUS
        # ====================================================

        if identity == "desconhecido":

            color = (
                0,
                0,
                255
            )


            label = (
                f"DESCONHECIDO "
                f"{score:.2f}"
            )

        else:

            color = (
                0,
                255,
                0
            )


            label = (
                f"{identity.upper()} "
                f"{score:.2f}"
            )


        # ====================================================
        # QUADRADO DO ROSTO
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # ====================================================
        # IDENTIDADE
        # ====================================================

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    y1 - 10,
                    25
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # MARGEM
        # ====================================================

        margem_texto = (
            f"Face {index + 1} | "
            f"Margem: {margin:.2f}"
        )


        cv2.putText(
            frame,
            margem_texto,
            (
                x1,
                min(
                    y2 + 20,
                    frame.shape[0] - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # FPS
    # ========================================================

    fps_counter += 1


    current_time = time.time()


    if (
        current_time
        -
        fps_timer
        >= 1
    ):

        fps = fps_counter

        fps_counter = 0

        fps_timer = current_time


    # ========================================================
    # INFORMAÇÕES DA INTERFACE
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (390, 105),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"InsightFace: {inference_ms:.1f} ms",
        (20, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # MOSTRAR
    # ========================================================

    cv2.imshow(
        "VisionAI V2 - Recognition Baseline",
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
# FINALIZAÇÃO
# ============================================================

camera.release()

cv2.destroyAllWindows()


print()
print("=" * 70)
print(
    "VISIONAI V2 - TESTE ENCERRADO"
)
print("=" * 70)
print()