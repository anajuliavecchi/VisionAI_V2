# ============================================================
# VISIONAI V2
# FACE DATABASE
#
# Responsável por carregar e validar os embeddings cadastrados.
# ============================================================

import os

import numpy as np


# ============================================================
# CONFIGURAÇÕES
# ============================================================

FACES_DIR = "faces"


# ============================================================
# CLASSE DO BANCO
# ============================================================

class FaceDatabase:

    def __init__(self, faces_dir=FACES_DIR):

        self.faces_dir = faces_dir

        self.database = {}


    # ========================================================
    # CARREGAR BANCO
    # ========================================================

    def load(self):

        if not os.path.isdir(self.faces_dir):

            raise RuntimeError(
                f"Pasta '{self.faces_dir}' não encontrada."
            )


        self.database = {}


        # ----------------------------------------------------
        # PESSOAS
        # ----------------------------------------------------

        for pessoa in sorted(
            os.listdir(self.faces_dir)
        ):

            pessoa_path = os.path.join(
                self.faces_dir,
                pessoa
            )


            if not os.path.isdir(pessoa_path):

                continue


            embeddings_dir = os.path.join(
                pessoa_path,
                "embeddings"
            )


            if not os.path.isdir(embeddings_dir):

                continue


            embeddings = []


            # ------------------------------------------------
            # EMBEDDINGS
            # ------------------------------------------------

            for filename in sorted(
                os.listdir(embeddings_dir)
            ):

                if not filename.lower().endswith(
                    ".npy"
                ):

                    continue


                caminho = os.path.join(
                    embeddings_dir,
                    filename
                )


                try:

                    embedding = np.load(
                        caminho
                    )


                    embedding = embedding.astype(
                        np.float32
                    )


                    # ----------------------------------------
                    # VALIDAR
                    # ----------------------------------------

                    if embedding.ndim != 1:

                        print(
                            f"⚠️ Ignorado: "
                            f"{caminho} "
                            f"(formato inválido)"
                        )

                        continue


                    norm = np.linalg.norm(
                        embedding
                    )


                    if norm == 0:

                        print(
                            f"⚠️ Ignorado: "
                            f"{caminho} "
                            f"(embedding inválido)"
                        )

                        continue


                    # ----------------------------------------
                    # NORMALIZAR
                    # ----------------------------------------

                    embedding = (
                        embedding / norm
                    )


                    embeddings.append(
                        embedding
                    )


                except Exception as erro:

                    print(
                        f"⚠️ Erro ao carregar "
                        f"{caminho}: {erro}"
                    )


            # ------------------------------------------------
            # ADICIONAR PESSOA
            # ------------------------------------------------

            if len(embeddings) > 0:

                self.database[pessoa] = embeddings


        # ====================================================
        # RESULTADO
        # ====================================================

        if len(self.database) == 0:

            raise RuntimeError(
                "Nenhuma pessoa cadastrada foi encontrada."
            )


        return self.database


    # ========================================================
    # INFORMAÇÕES
    # ========================================================

    def get_people(self):

        return list(
            self.database.keys()
        )


    def get_total_embeddings(self):

        return sum(
            len(embeddings)
            for embeddings in self.database.values()
        )


    def get_database(self):

        return self.database