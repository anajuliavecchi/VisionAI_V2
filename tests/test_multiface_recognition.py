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

sys.path.insert(
    0,
    ROOT_DIR
)


# ============================================================
# BIBLIOTECAS DO PROJETO
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
# CONFIGURAÇÕES DO TESTE
# ============================================================

# Quantidade mínima de evidências para estabilizar
HISTORY_SIZE = 5

# Quantidade de resultados iguais necessária
MIN_VOTES = 3

# Distância máxima entre o centro do rosto
# para considerar que é a mesma pessoa/track
MAX_TRACK_DISTANCE = 100

# Threshold inicial.
#
# IMPORTANTE:
# Este valor é apenas para TESTE.
# Não será considerado o threshold definitivo
# do VisionAI até fazermos o benchmark controlado.
RECOGNITION_THRESHOLD = 0.45


# ============================================================
# CARREGAR EMBEDDINGS
# ============================================================

def load_embeddings():

    database = {}

    faces_dir = os.path.join(
        ROOT_DIR,
        "faces"
    )

    print()
    print("=" * 70)
    print("CARREGANDO EMBEDDINGS")
    print("=" * 70)
    print()

    if not os.path.isdir(faces_dir):

        print(
            "ERRO: pasta faces/ não encontrada."
        )

        raise SystemExit


    people = sorted(
        [
            name
            for name in os.listdir(faces_dir)
            if os.path.isdir(
                os.path.join(
                    faces_dir,
                    name
                )
            )
        ]
    )


    for person in people:

        embedding_dir = os.path.join(
            faces_dir,
            person,
            "embeddings"
        )


        if not os.path.isdir(
            embedding_dir
        ):

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

                embedding = np.load(
                    file
                )


                embedding = np.asarray(
                    embedding,
                    dtype=np.float32
                ).flatten()


                norm = np.linalg.norm(
                    embedding
                )


                if norm == 0:

                    continue


                embedding = (
                    embedding / norm
                )


                embeddings.append(
                    embedding
                )


            except Exception as error:

                print(
                    f"Erro carregando "
                    f"{file}: {error}"
                )


        if embeddings:

            database[person] = (
                embeddings
            )


            print(
                f"{person.upper():12}"
                f" {len(embeddings)} embeddings"
            )


    print()

    print(
        f"Total de identidades: "
        f"{len(database)}"
    )

    print()

    if not database:

        print(
            "ERRO: nenhum embedding encontrado."
        )

        raise SystemExit


    return database


# ============================================================
# SIMILARIDADE DE COSSENO
# ============================================================

def cosine_similarity(
    embedding_a,
    embedding_b
):

    norm_a = np.linalg.norm(
        embedding_a
    )

    norm_b = np.linalg.norm(
        embedding_b
    )


    if norm_a == 0 or norm_b == 0:

        return 0.0


    return float(
        np.dot(
            embedding_a,
            embedding_b
        )
        /
        (
            norm_a
            *
            norm_b
        )
    )


# ============================================================
# RECONHECER UMA FACE
# ============================================================

def recognize_face(
    face,
    database
):

    if not hasattr(
        face,
        "embedding"
    ):

        return (
            "desconhecido",
            0.0,
            0.0
        )


    query_embedding = np.asarray(
        face.embedding,
        dtype=np.float32
    ).flatten()


    norm = np.linalg.norm(
        query_embedding
    )


    if norm == 0:

        return (
            "desconhecido",
            0.0,
            0.0
        )


    query_embedding = (
        query_embedding / norm
    )


    candidates = []


    for person, embeddings in (
        database.items()
    ):

        # ----------------------------------------------------
        # Comparamos contra TODOS os embeddings daquela pessoa.
        # ----------------------------------------------------

        scores = []


        for reference in embeddings:

            score = cosine_similarity(
                query_embedding,
                reference
            )

            scores.append(
                score
            )


        # Melhor evidência daquela identidade
        best_person_score = max(
            scores
        )


        candidates.append(
            (
                person,
                best_person_score
            )
        )


    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )


    if not candidates:

        return (
            "desconhecido",
            0.0,
            0.0
        )


    best_person = (
        candidates[0][0]
    )

    best_score = (
        candidates[0][1]
    )


    if len(candidates) >= 2:

        second_score = (
            candidates[1][1]
        )

    else:

        second_score = 0.0


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    if (
        best_score
        >= RECOGNITION_THRESHOLD
    ):

        identity = best_person

    else:

        identity = "desconhecido"


    return (
        identity,
        best_score,
        second_score
    )


# ============================================================
# TRACK
# ============================================================

class FaceTrack:

    def __init__(
        self,
        track_id,
        center
    ):

        self.track_id = track_id

        self.center = center

        self.history = deque(
            maxlen=HISTORY_SIZE
        )

        self.last_seen = time.time()


    def update(
        self,
        center,
        identity,
        score
    ):

        self.center = center

        self.last_seen = time.time()

        self.history.append(
            (
                identity,
                score
            )
        )


    def stable_identity(self):

        if not self.history:

            return (
                "ANALISANDO",
                0.0
            )


        votes = {}


        for identity, score in (
            self.history
        ):

            votes[identity] = (
                votes.get(
                    identity,
                    0
                )
                + 1
            )


        best_identity = max(
            votes,
            key=votes.get
        )


        vote_count = (
            votes[best_identity]
        )


        scores = [
            score
            for identity, score
            in self.history
            if identity == best_identity
        ]


        average_score = (
            np.mean(scores)
            if scores
            else 0.0
        )


        if vote_count >= MIN_VOTES:

            return (
                best_identity,
                float(average_score)
            )


        return (
            "ANALISANDO",
            float(average_score)
        )


# ============================================================
# GERENCIADOR DE TRACKS
# ============================================================

tracks = {}

next_track_id = 1


def get_track(
    center
):

    global next_track_id


    best_track = None

    best_distance = float(
        "inf"
    )


    for track in tracks.values():

        distance = np.linalg.norm(
            np.array(center)
            -
            np.array(track.center)
        )


        if (
            distance
            <
            best_distance
            and
            distance
            <= MAX_TRACK_DISTANCE
        ):

            best_distance = distance

            best_track = track


    if best_track is not None:

        return best_track


    new_track = FaceTrack(
        next_track_id,
        center
    )


    tracks[
        next_track_id
    ] = new_track


    next_track_id += 1


    return new_track


# ============================================================
# INSIGHTFACE
# ============================================================

database = load_embeddings()


print("=" * 70)
print("CARREGANDO INSIGHTFACE")
print("=" * 70)
print()


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
# LOOP PRINCIPAL
# ============================================================

while True:

    ret, frame = camera.read()


    if not ret:

        continue


    # --------------------------------------------------------
    # REDUÇÃO PARA DETECÇÃO
    # --------------------------------------------------------

    small_frame = cv2.resize(
        frame,
        DET_SIZE,
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # INSIGHTFACE
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
    # ESCALA
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


    # Tracks presentes neste frame
    current_track_ids = set()


    # ========================================================
    # PROCESSAR CADA FACE
    # ========================================================

    for face in faces:

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        x1, y1, x2, y2 = (
            face.bbox.astype(float)
        )


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


        # ----------------------------------------------------
        # LIMITAR À IMAGEM
        # ----------------------------------------------------

        x1 = max(
            0,
            min(
                x1,
                frame.shape[1] - 1
            )
        )


        x2 = max(
            0,
            min(
                x2,
                frame.shape[1] - 1
            )
        )


        y1 = max(
            0,
            min(
                y1,
                frame.shape[0] - 1
            )
        )


        y2 = max(
            0,
            min(
                y2,
                frame.shape[0] - 1
            )
        )


        # ----------------------------------------------------
        # CENTRO
        # ----------------------------------------------------

        center = (
            int(
                (x1 + x2) / 2
            ),
            int(
                (y1 + y2) / 2
            )
        )


        # ----------------------------------------------------
        # RECONHECIMENTO
        # ----------------------------------------------------

        (
            identity,
            score,
            second_score
        ) = recognize_face(
            face,
            database
        )


        # ----------------------------------------------------
        # TRACK
        # ----------------------------------------------------

        track = get_track(
            center
        )


        track.update(
            center,
            identity,
            score
        )


        current_track_ids.add(
            track.track_id
        )


        # ----------------------------------------------------
        # IDENTIDADE ESTABILIZADA
        # ----------------------------------------------------

        (
            stable_identity,
            stable_score
        ) = track.stable_identity()


        # ----------------------------------------------------
        # CORES
        # ----------------------------------------------------

        if stable_identity == (
            "desconhecido"
        ):

            color = (
                0,
                0,
                255
            )

            label = (
                "DESCONHECIDO"
            )


        elif stable_identity == (
            "ANALISANDO"
        ):

            color = (
                0,
                165,
                255
            )

            label = (
                "ANALISANDO"
            )


        else:

            color = (
                0,
                255,
                0
            )

            label = (
                stable_identity.upper()
            )


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
        # TEXTO
        # ====================================================

        text_y = max(
            y1 - 12,
            25
        )


        cv2.putText(
            frame,
            label,
            (
                x1,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score_text = (
            f"Score: {score:.2f}"
        )


        cv2.putText(
            frame,
            score_text,
            (
                x1,
                min(
                    y2 + 22,
                    frame.shape[0] - 40
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # TRACK
        # ----------------------------------------------------

        track_text = (
            f"ID: {track.track_id}"
        )


        cv2.putText(
            frame,
            track_text,
            (
                x1,
                min(
                    y2 + 40,
                    frame.shape[0] - 20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )


    # ========================================================
    # REMOVER TRACKS ANTIGOS
    # ========================================================

    now = time.time()


    expired_tracks = []


    for track_id, track in (
        tracks.items()
    ):

        if (
            now
            -
            track.last_seen
            >
            1.0
        ):

            expired_tracks.append(
                track_id
            )


    for track_id in expired_tracks:

        del tracks[
            track_id
        ]


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
    # PAINEL
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (430, 135),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Tracks: {len(tracks)}",
        (20, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"InsightFace: "
        f"{inference_ms:.1f} ms",
        (20, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )


    # ========================================================
    # MOSTRAR
    # ========================================================

    cv2.imshow(
        "VisionAI V2 - Multi-Face Recognition",
        frame
    )


    # ========================================================
    # SAIR
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
print("MULTI-FACE RECOGNITION ENCERRADO")
print("=" * 70)
print()