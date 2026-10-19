import sys
import os
import cv2
import time
import glob
import csv
import numpy as np
from collections import Counter


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


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TEST_DURATION = 20

RECOGNITION_THRESHOLD = 0.45

OUTPUT_FILE = os.path.join(
    ROOT_DIR,
    "benchmark_recognition.csv"
)


# ============================================================
# CARREGAR EMBEDDINGS
# ============================================================

def load_database():

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
                    f"Erro em {file}: "
                    f"{error}"
                )


        if embeddings:

            database[person] = embeddings

            print(
                f"{person.upper():12} "
                f"{len(embeddings)} embeddings"
            )


    print()

    print(
        f"Identidades carregadas: "
        f"{len(database)}"
    )

    print()

    return database


# ============================================================
# SIMILARIDADE
# ============================================================

def cosine_similarity(
    a,
    b
):

    norm_a = np.linalg.norm(a)

    norm_b = np.linalg.norm(b)


    if norm_a == 0 or norm_b == 0:

        return 0.0


    return float(
        np.dot(a, b)
        /
        (
            norm_a
            *
            norm_b
        )
    )


# ============================================================
# RECONHECIMENTO
# ============================================================

def recognize(
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


    embedding = np.asarray(
        face.embedding,
        dtype=np.float32
    ).flatten()


    norm = np.linalg.norm(
        embedding
    )


    if norm == 0:

        return (
            "desconhecido",
            0.0,
            0.0
        )


    embedding = (
        embedding / norm
    )


    candidates = []


    for person, references in (
        database.items()
    ):

        best_score = -1.0


        for reference in references:

            score = cosine_similarity(
                embedding,
                reference
            )


            if score > best_score:

                best_score = score


        candidates.append(
            (
                person,
                best_score
            )
        )


    if not candidates:

        return (
            "desconhecido",
            0.0,
            0.0
        )


    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )


    best_person = (
        candidates[0][0]
    )

    best_score = (
        candidates[0][1]
    )


    if len(candidates) > 1:

        second_score = (
            candidates[1][1]
        )

    else:

        second_score = 0.0


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
# RESULTADOS
# ============================================================

results = []


# ============================================================
# TESTE
# ============================================================

def run_test(
    app,
    camera,
    database,
    test_name,
    expected_identity
):

    print()
    print("=" * 70)
    print(
        f"TESTE: {test_name}"
    )
    print("=" * 70)
    print()

    if expected_identity == "desconhecido":

        print(
            "Esperado: DESCONHECIDO"
        )

    else:

        print(
            f"Esperado: "
            f"{expected_identity.upper()}"
        )


    print()
    print(
        f"Duração: {TEST_DURATION} segundos"
    )
    print()
    print(
        "Começando em 3..."
    )

    time.sleep(1)

    print("2...")

    time.sleep(1)

    print("1...")

    time.sleep(1)

    print("INICIADO!")
    print()


    start_time = time.perf_counter()

    total_frames = 0

    valid_frames = 0

    correct = 0

    incorrect = 0

    no_face = 0

    multiple_faces = 0

    scores = []

    margins = []

    identities = Counter()


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


        # ----------------------------------------------------
        # DETECÇÃO
        # ----------------------------------------------------

        small_frame = cv2.resize(
            frame,
            DET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )


        inference_start = (
            time.perf_counter()
        )


        faces = app.get(
            small_frame
        )


        inference_ms = (
            time.perf_counter()
            -
            inference_start
        ) * 1000


        # ----------------------------------------------------
        # SOMENTE UMA FACE
        #
        # Para o benchmark controlado,
        # ignoramos frames com 0 ou >1 faces.
        # ----------------------------------------------------

        if len(faces) == 0:

            no_face += 1

            cv2.putText(
                frame,
                "SEM ROSTO",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

            cv2.imshow(
                "VisionAI - Benchmark",
                frame
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                return None


            continue


        if len(faces) > 1:

            multiple_faces += 1

            cv2.putText(
                frame,
                "MULTIPLAS FACES - FRAME IGNORADO",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2
            )

            cv2.imshow(
                "VisionAI - Benchmark",
                frame
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):

                return None


            continue


        # ----------------------------------------------------
        # RECONHECIMENTO
        # ----------------------------------------------------

        face = faces[0]


        identity, score, second_score = (
            recognize(
                face,
                database
            )
        )


        valid_frames += 1


        scores.append(
            score
        )


        margins.append(
            score - second_score
        )


        identities[
            identity
        ] += 1


        # ----------------------------------------------------
        # CORRETO / INCORRETO
        # ----------------------------------------------------

        is_correct = (
            identity
            ==
            expected_identity
        )


        if is_correct:

            correct += 1

        else:

            incorrect += 1


        # ----------------------------------------------------
        # SALVAR RESULTADO
        # ----------------------------------------------------

        results.append(
            {
                "teste": test_name,
                "esperado": expected_identity,
                "obtido": identity,
                "score": score,
                "segundo_score": second_score,
                "margem": score - second_score,
                "inferencia_ms": inference_ms
            }
        )


        # ----------------------------------------------------
        # COORDENADAS
        # ----------------------------------------------------

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
        # COR
        # ----------------------------------------------------

        if is_correct:

            color = (
                0,
                255,
                0
            )

        else:

            color = (
                0,
                0,
                255
            )


        # ----------------------------------------------------
        # CAIXA
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # ----------------------------------------------------
        # IDENTIDADE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            identity.upper(),
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


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Score: {score:.3f}",
            (
                x1,
                min(
                    y2 + 25,
                    frame.shape[0] - 20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # INFORMAÇÕES
        # ----------------------------------------------------

        remaining = (
            TEST_DURATION
            -
            elapsed
        )


        cv2.rectangle(
            frame,
            (10, 10),
            (430, 105),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            f"Teste: {test_name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Esperado: "
            f"{expected_identity.upper()}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Tempo: "
            f"{remaining:.1f}s",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "VisionAI - Benchmark",
            frame
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):

            return None


    # ========================================================
    # MÉTRICAS DO TESTE
    # ========================================================

    accuracy = (
        correct / valid_frames * 100
        if valid_frames > 0
        else 0
    )


    avg_score = (
        np.mean(scores)
        if scores
        else 0
    )


    median_score = (
        np.median(scores)
        if scores
        else 0
    )


    p95_score = (
        np.percentile(scores, 95)
        if scores
        else 0
    )


    avg_margin = (
        np.mean(margins)
        if margins
        else 0
    )


    print()
    print(
        f"RESULTADO: {test_name}"
    )
    print("-" * 70)

    print(
        f"Frames capturados: "
        f"{total_frames}"
    )

    print(
        f"Frames válidos: "
        f"{valid_frames}"
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
        f"Corretos: "
        f"{correct}"
    )

    print(
        f"Incorretos: "
        f"{incorrect}"
    )

    print(
        f"Taxa de acerto: "
        f"{accuracy:.2f}%"
    )

    print()

    print(
        "SCORES"
    )

    print(
        f"  Média:   {avg_score:.4f}"
    )

    print(
        f"  Mediana: {median_score:.4f}"
    )

    print(
        f"  P95:     {p95_score:.4f}"
    )

    print()

    print(
        "MARGEM"
    )

    print(
        f"  Média:   {avg_margin:.4f}"
    )

    print()

    print(
        "IDENTIDADES"
    )

    for identity, count in (
        identities.most_common()
    ):

        percentage = (
            count
            /
            valid_frames
            *
            100
            if valid_frames > 0
            else 0
        )


        print(
            f"  {identity:15}"
            f"{count:5}"
            f" ({percentage:5.1f}%)"
        )


    return {
        "teste": test_name,
        "esperado": expected_identity,
        "frames": total_frames,
        "validos": valid_frames,
        "corretos": correct,
        "incorretos": incorrect,
        "accuracy": accuracy,
        "no_face": no_face,
        "multiple_faces": multiple_faces,
        "avg_score": avg_score,
        "median_score": median_score,
        "p95_score": p95_score,
        "avg_margin": avg_margin
    }


# ============================================================
# SALVAR CSV
# ============================================================

def save_csv():

    if not results:

        return


    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "teste",
                "esperado",
                "obtido",
                "score",
                "segundo_score",
                "margem",
                "inferencia_ms"
            ]
        )


        writer.writeheader()


        writer.writerows(
            results
        )


# ============================================================
# MAIN
# ============================================================

def main():

    database = load_database()


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

        raise SystemExit


    # ========================================================
    # INSTRUÇÕES
    # ========================================================

    print()
    print("=" * 70)
    print("VISIONAI V2 - BENCHMARK CONTROLADO")
    print("=" * 70)
    print()

    print(
        "Cada teste terá 20 segundos."
    )

    print(
        "Durante o teste, mantenha APENAS"
    )

    print(
        "a pessoa esperada na frente da câmera."
    )

    print()
    print(
        "Q = interromper teste"
    )

    print()


    # ========================================================
    # TESTE 1
    # ========================================================

    result_1 = run_test(
        app,
        camera,
        database,
        "ANA_NORMAL",
        "ana"
    )


    if result_1 is None:

        camera.release()

        cv2.destroyAllWindows()

        return


    # ========================================================
    # TESTE 2
    # ========================================================

    result_2 = run_test(
        app,
        camera,
        database,
        "ANA_VARIADA",
        "ana"
    )


    if result_2 is None:

        camera.release()

        cv2.destroyAllWindows()

        return


    # ========================================================
    # TESTE 3
    # ========================================================

    result_3 = run_test(
        app,
        camera,
        database,
        "DESCONHECIDA_NORMAL",
        "desconhecido"
    )


    if result_3 is None:

        camera.release()

        cv2.destroyAllWindows()

        return


    # ========================================================
    # TESTE 4
    # ========================================================

    result_4 = run_test(
        app,
        camera,
        database,
        "DESCONHECIDA_VARIADA",
        "desconhecido"
    )


    if result_4 is None:

        camera.release()

        cv2.destroyAllWindows()

        return


    # ========================================================
    # ENCERRAR
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()


    # ========================================================
    # SALVAR
    # ========================================================

    save_csv()


    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    test_results = [
        result_1,
        result_2,
        result_3,
        result_4
    ]


    total_valid = sum(
        result["validos"]
        for result in test_results
    )


    total_correct = sum(
        result["corretos"]
        for result in test_results
    )


    total_incorrect = sum(
        result["incorretos"]
        for result in test_results
    )


    overall_accuracy = (
        total_correct
        /
        total_valid
        *
        100
        if total_valid > 0
        else 0
    )


    # ========================================================
    # FALSO POSITIVO
    # ========================================================

    unknown_results = [
        result
        for result in test_results
        if result["esperado"]
        ==
        "desconhecido"
    ]


    unknown_valid = sum(
        result["validos"]
        for result in unknown_results
    )


    false_accepts = sum(
        result["incorretos"]
        for result in unknown_results
    )


    false_accept_rate = (
        false_accepts
        /
        unknown_valid
        *
        100
        if unknown_valid > 0
        else 0
    )


    # ========================================================
    # FALSO NEGATIVO
    # ========================================================

    known_results = [
        result
        for result in test_results
        if result["esperado"]
        !=
        "desconhecido"
    ]


    known_valid = sum(
        result["validos"]
        for result in known_results
    )


    false_rejects = sum(
        result["incorretos"]
        for result in known_results
    )


    false_reject_rate = (
        false_rejects
        /
        known_valid
        *
        100
        if known_valid > 0
        else 0
    )


    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("              RESULTADO FINAL")
    print("=" * 70)
    print()

    print(
        f"Frames válidos totais: "
        f"{total_valid}"
    )

    print(
        f"Decisões corretas: "
        f"{total_correct}"
    )

    print(
        f"Decisões incorretas: "
        f"{total_incorrect}"
    )

    print()

    print(
        f"ACURÁCIA GERAL: "
        f"{overall_accuracy:.2f}%"
    )

    print()

    print(
        "FALSO POSITIVO"
    )

    print(
        "Desconhecido reconhecido como integrante:"
    )

    print(
        f"  {false_accepts} ocorrências"
    )

    print(
        f"  Taxa: "
        f"{false_accept_rate:.2f}%"
    )

    print()

    print(
        "FALSO NEGATIVO"
    )

    print(
        "Integrante reconhecido como desconhecido:"
    )

    print(
        f"  {false_rejects} ocorrências"
    )

    print(
        f"  Taxa: "
        f"{false_reject_rate:.2f}%"
    )

    print()

    print(
        "Arquivo detalhado:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print("=" * 70)
    print("BENCHMARK FINALIZADO")
    print("=" * 70)
    print()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()