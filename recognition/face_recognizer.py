# ============================================================
# VISIONAI V2
# FACE RECOGNIZER
#
# Responsável exclusivamente pela comparação de embeddings.
#
# Não controla câmera.
# Não controla InsightFace.
# Não desenha interface.
#
# Entrada:
#   embedding de um rosto
#
# Saída:
#   pessoa + score
# ============================================================

import numpy as np


# ============================================================
# RECONHECEDOR
# ============================================================

class FaceRecognizer:

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


        norm = np.linalg.norm(
            embedding
        )


        if norm == 0:

            return None


        return (
            embedding / norm
        )


    # ========================================================
    # SIMILARIDADE COSSENO
    # ========================================================

    @staticmethod
    def cosine_similarity(
        embedding1,
        embedding2
    ):

        embedding1 = (
            FaceRecognizer.normalize(
                embedding1
            )
        )


        embedding2 = (
            FaceRecognizer.normalize(
                embedding2
            )
        )


        if (
            embedding1 is None
            or
            embedding2 is None
        ):

            return -1.0


        return float(
            np.dot(
                embedding1,
                embedding2
            )
        )


    # ========================================================
    # IDENTIFICAR
    # ========================================================

    def identify(
        self,
        embedding
    ):

        embedding = self.normalize(
            embedding
        )


        if embedding is None:

            return {
                "name": "desconhecido",
                "score": -1.0,
                "second_score": -1.0,
                "margin": 0.0
            }


        melhor_pessoa = (
            "desconhecido"
        )

        melhor_score = -1.0

        segundo_melhor_score = -1.0


        # ====================================================
        # COMPARAR COM TODAS AS PESSOAS
        # ====================================================

        for pessoa, embeddings in (
            self.database.items()
        ):

            melhor_score_pessoa = -1.0


            for cadastro_embedding in embeddings:

                score = (
                    self.cosine_similarity(
                        embedding,
                        cadastro_embedding
                    )
                )


                if score > melhor_score_pessoa:

                    melhor_score_pessoa = (
                        score
                    )


            # ------------------------------------------------
            # MELHOR PESSOA
            # ------------------------------------------------

            if (
                melhor_score_pessoa
                >
                melhor_score
            ):

                segundo_melhor_score = (
                    melhor_score
                )

                melhor_score = (
                    melhor_score_pessoa
                )

                melhor_pessoa = (
                    pessoa
                )

            elif (
                melhor_score_pessoa
                >
                segundo_melhor_score
            ):

                segundo_melhor_score = (
                    melhor_score_pessoa
                )


        # ====================================================
        # MARGEM
        # ====================================================

        margin = (
            melhor_score
            -
            segundo_melhor_score
        )


        # ====================================================
        # DECISÃO
        # ====================================================

        if (
            melhor_score
            >=
            self.threshold
        ):

            nome_final = (
                melhor_pessoa
            )

        else:

            nome_final = (
                "desconhecido"
            )


        return {
            "name": nome_final,
            "score": melhor_score,
            "second_score": segundo_melhor_score,
            "margin": margin
        }