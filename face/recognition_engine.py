import numpy as np


# ============================================================
# VISIONAI V2
# MOTOR DE RECONHECIMENTO FACIAL
# ============================================================


class RecognitionEngine:

    def __init__(
        self,
        database,
        threshold=0.40
    ):

        self.database = database

        self.threshold = threshold


    # ========================================================
    # NORMALIZAÇÃO
    # ========================================================

    @staticmethod
    def normalize(
        embedding
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )


        norma = np.linalg.norm(
            embedding
        )


        if norma == 0:

            return None


        return (
            embedding / norma
        )


    # ========================================================
    # SIMILARIDADE DE COSSENO
    # ========================================================

    @staticmethod
    def cosine_similarity(
        embedding1,
        embedding2
    ):

        return float(
            np.dot(
                embedding1,
                embedding2
            )
        )


    # ========================================================
    # RECONHECER UMA FACE
    # ========================================================

    def recognize(
        self,
        embedding
    ):

        # ----------------------------------------------------
        # NORMALIZAR EMBEDDING DA CÂMERA
        # ----------------------------------------------------

        embedding = self.normalize(
            embedding
        )


        if embedding is None:

            return {

                "identity":
                    "desconhecido",

                "score":
                    0.0,

                "second_identity":
                    None,

                "second_score":
                    0.0,

                "margin":
                    0.0
            }


        candidates = []


        # ====================================================
        # COMPARAR COM TODAS AS PESSOAS
        # ====================================================

        for pessoa, embeddings in (
            self.database.items()
        ):

            melhor_score_pessoa = -1.0


            # ------------------------------------------------
            # COMPARAR COM TODAS AS AMOSTRAS
            # ------------------------------------------------

            for reference in embeddings:

                score = (
                    self.cosine_similarity(
                        embedding,
                        reference
                    )
                )


                if (
                    score
                    >
                    melhor_score_pessoa
                ):

                    melhor_score_pessoa = (
                        score
                    )


            candidates.append(
                (
                    pessoa,
                    melhor_score_pessoa
                )
            )


        # ====================================================
        # NENHUM CANDIDATO
        # ====================================================

        if not candidates:

            return {

                "identity":
                    "desconhecido",

                "score":
                    0.0,

                "second_identity":
                    None,

                "second_score":
                    0.0,

                "margin":
                    0.0
            }


        # ====================================================
        # ORDENAR MELHORES RESULTADOS
        # ====================================================

        candidates.sort(
            key=lambda item: item[1],
            reverse=True
        )


        # Primeiro candidato
        best_identity = (
            candidates[0][0]
        )

        best_score = (
            candidates[0][1]
        )


        # Segundo candidato
        if len(candidates) > 1:

            second_identity = (
                candidates[1][0]
            )

            second_score = (
                candidates[1][1]
            )

        else:

            second_identity = None

            second_score = 0.0


        # ====================================================
        # MARGEM
        # ====================================================

        margin = (
            best_score
            -
            second_score
        )


        # ====================================================
        # DECISÃO
        # ====================================================

        if (
            best_score
            >=
            self.threshold
        ):

            identity = (
                best_identity
            )

        else:

            identity = (
                "desconhecido"
            )


        # ====================================================
        # RESULTADO
        # ====================================================

        return {

            "identity":
                identity,

            "score":
                float(
                    best_score
                ),

            "second_identity":
                second_identity,

            "second_score":
                float(
                    second_score
                ),

            "margin":
                float(
                    margin
                )
        }