import os
import shutil
from pathlib import Path

import numpy as np


class FaceDatabase:
    """Banco de embeddings faciais persistidos em faces/<nome>/embeddings."""

    def __init__(self, faces_dir):
        self.faces_dir = os.path.abspath(os.fspath(faces_dir))
        self.database = {}

    def load(self):
        if not os.path.isdir(self.faces_dir):
            raise FileNotFoundError(f"Pasta não encontrada: {self.faces_dir}")

        self.database.clear()
        for pessoa in sorted(os.listdir(self.faces_dir)):
            pessoa_path = os.path.join(self.faces_dir, pessoa)
            if not os.path.isdir(pessoa_path):
                continue

            embeddings_dir = os.path.join(pessoa_path, "embeddings")
            if not os.path.isdir(embeddings_dir):
                continue

            embeddings = []
            for filename in sorted(os.listdir(embeddings_dir)):
                if not filename.lower().endswith(".npy"):
                    continue
                caminho = os.path.join(embeddings_dir, filename)
                try:
                    embedding = np.load(caminho, allow_pickle=False).astype(np.float32)
                    norma = np.linalg.norm(embedding)
                    if norma == 0 or not np.isfinite(norma):
                        print(f"[DATABASE] Embedding inválido ignorado: {caminho}")
                        continue
                    embeddings.append(embedding / norma)
                except Exception as erro:
                    print(f"[DATABASE] Erro carregando {caminho}: {erro}")

            if embeddings:
                self.database[pessoa] = embeddings

    def people(self):
        return sorted(self.database.keys(), key=str.casefold)

    def get_embeddings(self, pessoa):
        return self.database.get(pessoa, [])

    def total_embeddings(self):
        return sum(len(embeddings) for embeddings in self.database.values())

    def delete_person(self, pessoa):
        """Remove os embeddings da memória e a pasta persistente da pessoa."""
        if not isinstance(pessoa, str) or not pessoa.strip():
            raise ValueError("Nome de pessoa inválido.")

        pessoa = pessoa.strip()
        if pessoa not in self.database:
            raise KeyError(f'O cadastro "{pessoa}" não existe no banco facial.')

        root = Path(self.faces_dir).resolve()
        target = (root / pessoa).resolve()
        if target == root or root not in target.parents:
            raise ValueError("Caminho de cadastro inválido; exclusão bloqueada.")

        # Remover primeiro do disco: se falhar, a pessoa continua reconhecível.
        if target.exists():
            if not target.is_dir():
                raise ValueError("O caminho do cadastro não é uma pasta.")
            shutil.rmtree(target)

        # A API e o reconhecedor usam a mesma referência para este dicionário.
        del self.database[pessoa]
        return {
            "person": pessoa,
            "total": len(self.database),
            "embeddings": self.total_embeddings(),
        }
