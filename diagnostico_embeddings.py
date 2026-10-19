
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"

print("=== DIAGNÓSTICO DE EMBEDDINGS ===")
print(f"Diretório analisado: {FACES_DIR}")
print()

if not FACES_DIR.exists():
    print("ERRO: pasta faces não encontrada.")
    raise SystemExit(1)

total = 0
dimensoes = {}
erros = []

for arquivo in sorted(FACES_DIR.glob("*/embeddings/*.npy")):
    total += 1

    try:
        vetor = np.load(arquivo, allow_pickle=False)
        formato = tuple(vetor.shape)
        dimensoes[formato] = dimensoes.get(formato, 0) + 1

        valido = (
            vetor.size > 0
            and np.issubdtype(vetor.dtype, np.number)
            and np.isfinite(vetor).all()
            and np.linalg.norm(vetor.astype(np.float64)) > 0
        )

        if not valido:
            erros.append(f"{arquivo.relative_to(FACES_DIR)}: vetor inválido")

    except Exception as exc:
        erros.append(
            f"{arquivo.relative_to(FACES_DIR)}: erro ao ler ({type(exc).__name__})"
        )

print(f"Total de embeddings encontrados: {total}")
print("Dimensões encontradas:")

for formato, quantidade in sorted(dimensoes.items()):
    print(f"  {formato}: {quantidade} arquivo(s)")

print(f"Embeddings inválidos: {len(erros)}")

for erro in erros:
    print(f"  - {erro}")

if total == 0:
    print("ATENÇÃO: nenhum embedding encontrado.")
