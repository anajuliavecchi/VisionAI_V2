import sys
import os
import cv2
import time
import glob
import numpy as np
from collections import deque


# ============================================================
# RAIZ DO PROJETO
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, ROOT_DIR)


# ============================================================
# IMPORTS
# ============================================================

from insightface.app import FaceAnalysis

from config import (
    MODEL_NAME,
    CAMERA_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    DET_SIZE
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

THRESHOLD = 0.45

HISTORY_SIZE = 7

MIN_VOTES = 4

TEST_DURATION = 60


# ============================================================
# CARREGAR EMBEDDINGS
# ============================================================

def load_database():

    database = {}

    faces_dir = os.path.join(
        ROOT_DIR,
        "faces"
    )

    for person in sorted(
        os.listdir(faces_dir)
    ):

        person_dir = os.path.join(
            faces_dir,
            person
        )

        if not os.path.isdir(person_dir):
            continue

        embedding_dir = os.path.join(
            person_dir,
            "embeddings"
        )

        if not os.path.isdir(embedding_dir):
            continue

        files = sorted(
            glob.glob(
                os.path.join(
                    embedding_dir,
                    "*.npy"
                )
            )
        )

        embeddings = []

        for file in files:

            try:

                emb = np.load(
                    file
                ).astype(
                    np.float32
                ).flatten()

                norm = np.linalg.norm(emb)

                if norm == 0:
                    continue

                emb /= norm

                embeddings.append(emb)

            except Exception as error:

                print(
                    f"Erro em {file}: {error}"
                )

        if embeddings:

            database[person] = embeddings

            print(
                f"{person.upper():12} "
                f"{len(embeddings)} embeddings"
            )

    return database


# ============================================================
# SIMILARIDADE
# ============================================================

def similarity(a, b):

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


# ============================================================
# RECONHECIMENTO
# ============================================================

def recognize(face, database):

    embedding = np.asarray(
        face.embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        return (
            "desconhecido",
            0.0
        )

    embedding /= norm

    best_identity = "desconhecido"

    best_score = -1.0

    for person, references in database.items():

        for reference in references:

            score = similarity(
                embedding,
                reference
            )

            if score > best_score:

                best_score = score

                best_identity = person


    if best_score >= THRESHOLD:

        return (
            best_identity,
            best_score
        )

    return (
        "desconhecido",
        best_score
    )


# ============================================================
# HISTÓRICO TEMPORAL
# ============================================================

history = deque(
    maxlen=HISTORY_SIZE
)


def temporal_decision(
    identity,
    score
):

    history.append(
        (
            identity,
            score
        )
    )

    if len(history) < MIN_VOTES:

        return (
            "ANALISANDO",
            0.0
        )


    votes = {}

    for item_identity, item_score in history:

        votes[item_identity] = (
            votes.get(
                item_identity,
                0
            )
            + 1
        )


    winning_identity = max(
        votes,
        key=votes.get
    )


    winning_votes = votes[
        winning_identity
    ]


    if winning_votes < MIN_VOTES:

        return (
            "ANALISANDO",
            0.0
        )


    scores = [
        item_score
        for item_identity, item_score
        in history
        if item_identity
        ==
        winning_identity
    ]


    average_score = float(
        np.mean(scores)
    )


    return (
        winning_identity,
        average_score
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VISIONAI V2 - TESTE DE ESTABILIDADE TEMPORAL")
    print("=" * 70)
    print()


    database = load_database()


    print()
    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Histórico: "
        f"{HISTORY_SIZE} frames"
    )

    print(
        f"Votos mínimos: "
        f"{MIN_VOTES}"
    )

    print()


    # ========================================================
    # INSIGHTFACE
    # ========================================================

    print(
        "Carregando InsightFace..."
    )


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


    print(
        "InsightFace carregado."
    )


    # ========================================================
    # CÂMERA
    # ========================================================

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

        return


    print()
    print(
        "Câmera iniciada."
    )

    print(
        "Fique sozinho na frente da câmera."
    )

    print(
        "Faça movimentos naturais."
    )

    print(
        "Q = sair"
    )

    print()


    # ========================================================
    # MÉTRICAS
    # ========================================================

    raw_correct = 0

    raw_unknown = 0

    temporal_ana = 0

    temporal_unknown = 0

    total_valid = 0

    total_frames = 0

    no_face = 0

    multiple_faces = 0


    score_history = []

    start_time = time.perf_counter()


    # ========================================================
    # LOOP
    # ========================================================

    while True:

        elapsed = (
            time.perf_counter()
            -
            start_time
        )


        if elapsed >= TEST_DURATION:

            break


        ret, frame = camera.read()


        if not ret:

            continue


        total_frames += 1


        small_frame = cv2.resize(
            frame,
            DET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )


        faces = app.get(
            small_frame
        )


        # ----------------------------------------------------
        # SEM ROSTO
        # ----------------------------------------------------

        if len(faces) == 0:

            no_face += 1

            continue


        # ----------------------------------------------------
        # MULTIFACE
        # ----------------------------------------------------

        if len(faces) > 1:

            multiple_faces += 1

            continue


        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        face = faces[0]


        raw_identity, score = recognize(
            face,
            database
        )


        total_valid += 1

        score_history.append(
            score
        )


        # ----------------------------------------------------
        # DECISÃO TEMPORAL
        # ----------------------------------------------------

        stable_identity, stable_score = (
            temporal_decision(
                raw_identity,
                score
            )
        )


        # ====================================================
        # MÉTRICAS
        # ====================================================

        if raw_identity == "ana":

            raw_correct += 1

        else:

            raw_unknown += 1


        if stable_identity == "ana":

            temporal_ana += 1

        elif stable_identity == "desconhecido":

            temporal_unknown += 1


        # ====================================================
        # BBOX
        # ====================================================

        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )


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


        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)

        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)


        # ====================================================
        # COR
        # ====================================================

        if stable_identity == "ana":

            color = (
                0,
                255,
                0
            )

            label = "ANA"


        elif stable_identity == "desconhecido":

            color = (
                0,
                0,
                255
            )

            label = "DESCONHECIDO"


        else:

            color = (
                0,
                165,
                255
            )

            label = "ANALISANDO"


        # ====================================================
        # CAIXA
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # ====================================================
        # LABEL
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
            0.7,
            color,
            2,
            cv2.LINE_AA
        )


        # ====================================================
        # SCORE BRUTO
        # ====================================================

        cv2.putText(
            frame,
            f"Score: {score:.3f}",
            (
                x1,
                min(
                    y2 + 22,
                    frame.shape[0] - 40
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # SCORE ESTABILIZADO
        # ====================================================

        cv2.putText(
            frame,
            f"Estavel: {stable_score:.3f}",
            (
                x1,
                min(
                    y2 + 42,
                    frame.shape[0] - 20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # PAINEL
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (440, 120),
            (0, 0, 0),
            -1
        )


        remaining = (
            TEST_DURATION
            -
            elapsed
        )


        cv2.putText(
            frame,
            f"Tempo: {remaining:.1f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Frames: {total_valid}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Bruto Ana: {raw_correct}",
            (20, 89),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Estavel Ana: {temporal_ana}",
            (20, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "VisionAI - Temporal Recognition",
            frame
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):

            break


    # ========================================================
    # ENCERRAMENTO
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


    # ========================================================
    # RESULTADOS
    # ========================================================

    raw_accuracy = (
        raw_correct
        /
        total_valid
        *
        100
        if total_valid > 0
        else 0
    )


    temporal_accuracy = (
        temporal_ana
        /
        total_valid
        *
        100
        if total_valid > 0
        else 0
    )


    avg_score = (
        np.mean(score_history)
        if score_history
        else 0
    )


    median_score = (
        np.median(score_history)
        if score_history
        else 0
    )


    print()
    print("=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print()

    print(
        f"Tempo de teste: "
        f"{TEST_DURATION}s"
    )

    print(
        f"Frames capturados: "
        f"{total_frames}"
    )

    print(
        f"Frames válidos: "
        f"{total_valid}"
    )

    print(
        f"Sem rosto: "
        f"{no_face}"
    )

    print(
        f"Múltiplas faces: "
        f"{multiple_faces}"
    )

    print()

    print(
        "DECISÃO FRAME A FRAME"
    )

    print(
        f"Ana: "
        f"{raw_correct}"
    )

    print(
        f"Desconhecido: "
        f"{raw_unknown}"
    )

    print(
        f"Taxa Ana: "
        f"{raw_accuracy:.2f}%"
    )

    print()

    print(
        "DECISÃO TEMPORAL"
    )

    print(
        f"Ana estabilizada: "
        f"{temporal_ana}"
    )

    print(
        f"Desconhecido estabilizado: "
        f"{temporal_unknown}"
    )

    print(
        f"Taxa Ana estabilizada: "
        f"{temporal_accuracy:.2f}%"
    )

    print()

    print(
        "SCORES"
    )

    print(
        f"Média: "
        f"{avg_score:.4f}"
    )

    print(
        f"Mediana: "
        f"{median_score:.4f}"
    )

    print()

    print("=" * 70)
    print(
        "TESTE TEMPORAL FINALIZADO"
    )
    print("=" * 70)
    print()


if __name__ == "__main__":

    main()