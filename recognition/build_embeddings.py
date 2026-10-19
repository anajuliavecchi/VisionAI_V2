# ============================================================
# VISIONAI V2
# BUILD EMBEDDINGS
#
# Gera novamente os embeddings faciais a partir das fotos
# existentes no banco de faces.
#
# Objetivos:
# - detectar cada rosto novamente;
# - rejeitar imagens inválidas;
# - gerar embeddings com o mesmo modelo da V2;
# - normalizar os embeddings;
# - substituir os embeddings antigos SOMENTE na V2;
# - gerar um relatório completo do processamento.
# ============================================================


# ============================================================
# BIBLIOTECAS
# ============================================================

import os
import shutil
import time

import cv2
import numpy as np

from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURAÇÕES
# ============================================================

FACES_DIR = "faces"

MODEL_NAME = "buffalo_sc"

DET_SIZE = (320, 320)

# Extensões aceitas
EXTENSOES_IMAGEM = (
    ".jpg",
    ".jpeg",
    ".png"
)


# ============================================================
# INÍCIO
# ============================================================

print()
print("=" * 70)
print("             VISIONAI V2 | BUILD EMBEDDINGS")
print("=" * 70)
print()

print(
    f"Pasta de faces: {FACES_DIR}"
)

print(
    f"Modelo: {MODEL_NAME}"
)

print(
    f"Detector: {DET_SIZE[0]}x{DET_SIZE[1]}"
)

print()


# ============================================================
# VERIFICAR PASTA
# ============================================================

if not os.path.isdir(FACES_DIR):

    print(
        f"ERRO: a pasta '{FACES_DIR}' não existe."
    )

    print(
        "Execute este programa a partir da raiz da VisionAI_V2."
    )

    exit(1)


# ============================================================
# CARREGAR INSIGHTFACE
# ============================================================

print(
    "[1/4] Carregando InsightFace..."
)

inicio_modelo = time.perf_counter()


try:

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

except Exception as erro:

    print()
    print(
        "ERRO ao carregar InsightFace:"
    )

    print(
        erro
    )

    exit(1)


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
# ENCONTRAR PESSOAS
# ============================================================

print(
    "[2/4] Procurando pessoas cadastradas..."
)

pessoas = []


for nome in sorted(
    os.listdir(FACES_DIR)
):

    caminho = os.path.join(
        FACES_DIR,
        nome
    )


    if not os.path.isdir(caminho):

        continue


    pessoas.append(nome)


print(
    f"Pessoas encontradas: "
    f"{len(pessoas)}"
)

print()


if len(pessoas) == 0:

    print(
        "ERRO: nenhuma pessoa encontrada."
    )

    exit(1)


# ============================================================
# ESTATÍSTICAS GERAIS
# ============================================================

total_fotos = 0

total_sucesso = 0

total_sem_rosto = 0

total_multiplos_rostos = 0

total_erro_imagem = 0

total_embedding_invalido = 0


relatorio = {}


# ============================================================
# PROCESSAR CADA PESSOA
# ============================================================

for numero_pessoa, pessoa in enumerate(
    pessoas,
    start=1
):

    print("=" * 70)

    print(
        f"PESSOA {numero_pessoa}/{len(pessoas)}: "
        f"{pessoa.upper()}"
    )

    print("=" * 70)


    pessoa_dir = os.path.join(
        FACES_DIR,
        pessoa
    )


    embeddings_dir = os.path.join(
        pessoa_dir,
        "embeddings"
    )


    # ========================================================
    # LOCALIZAR FOTOS
    # ========================================================

    fotos = []


    for filename in sorted(
        os.listdir(pessoa_dir)
    ):

        caminho = os.path.join(
            pessoa_dir,
            filename
        )


        if not os.path.isfile(caminho):

            continue


        if not filename.lower().endswith(
            EXTENSOES_IMAGEM
        ):

            continue


        fotos.append(filename)


    print(
        f"Fotos encontradas: "
        f"{len(fotos)}"
    )

    print()


    total_fotos += len(fotos)


    if len(fotos) == 0:

        print(
            "⚠️ Nenhuma foto encontrada."
        )

        relatorio[pessoa] = {
            "total": 0,
            "sucesso": 0,
            "sem_rosto": 0,
            "multiplos": 0,
            "erro_imagem": 0,
            "embedding_invalido": 0
        }

        continue


    # ========================================================
    # PREPARAR PASTA DE EMBEDDINGS
    # ========================================================

    # Remove os embeddings antigos SOMENTE da V2
    if os.path.isdir(embeddings_dir):

        shutil.rmtree(
            embeddings_dir
        )


    os.makedirs(
        embeddings_dir,
        exist_ok=True
    )


    # ========================================================
    # ESTATÍSTICAS DA PESSOA
    # ========================================================

    sucesso_pessoa = 0

    sem_rosto_pessoa = 0

    multiplos_pessoa = 0

    erro_imagem_pessoa = 0

    embedding_invalido_pessoa = 0


    # ========================================================
    # PROCESSAR FOTOS
    # ========================================================

    for numero_foto, filename in enumerate(
        fotos,
        start=1
    ):

        caminho = os.path.join(
            pessoa_dir,
            filename
        )


        print(
            f"[{numero_foto:02d}/{len(fotos):02d}] "
            f"{filename}"
        )


        # ----------------------------------------------------
        # ABRIR IMAGEM
        # ----------------------------------------------------

        image = cv2.imread(
            caminho
        )


        if image is None:

            print(
                "   ❌ Não foi possível abrir."
            )

            erro_imagem_pessoa += 1

            total_erro_imagem += 1

            continue


        # ----------------------------------------------------
        # REDUZIR IMAGEM
        # ----------------------------------------------------

        small_image = cv2.resize(
            image,
            DET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # DETECÇÃO
        # ----------------------------------------------------

        try:

            faces = app.get(
                small_image
            )

        except Exception as erro:

            print(
                f"   ❌ Erro no InsightFace: "
                f"{erro}"
            )

            erro_imagem_pessoa += 1

            total_erro_imagem += 1

            continue


        # ----------------------------------------------------
        # NENHUM ROSTO
        # ----------------------------------------------------

        if len(faces) == 0:

            print(
                "   ❌ Nenhum rosto detectado."
            )

            sem_rosto_pessoa += 1

            total_sem_rosto += 1

            continue


        # ----------------------------------------------------
        # MAIS DE UM ROSTO
        # ----------------------------------------------------

        if len(faces) > 1:

            print(
                f"   ❌ {len(faces)} rostos detectados."
            )

            print(
                "      Foto rejeitada."
            )

            multiplos_pessoa += 1

            total_multiplos_rostos += 1

            continue


        # ----------------------------------------------------
        # ROSTO ÚNICO
        # ----------------------------------------------------

        face = faces[0]


        detection_score = float(
            face.det_score
        )


        print(
            f"   Detecção: "
            f"{detection_score:.3f}"
        )


        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        embedding = face.embedding


        if embedding is None:

            print(
                "   ❌ Embedding inválido."
            )

            embedding_invalido_pessoa += 1

            total_embedding_invalido += 1

            continue


        # ----------------------------------------------------
        # CONVERTER PARA FLOAT32
        # ----------------------------------------------------

        embedding = embedding.astype(
            np.float32
        )


        # ----------------------------------------------------
        # NORMALIZAR
        # ----------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )


        if norm == 0:

            print(
                "   ❌ Norma do embedding é zero."
            )

            embedding_invalido_pessoa += 1

            total_embedding_invalido += 1

            continue


        embedding = (
            embedding / norm
        )


        # ----------------------------------------------------
        # SALVAR
        # ----------------------------------------------------

        nome_saida = (
            os.path.splitext(filename)[0]
            + ".npy"
        )


        caminho_saida = os.path.join(
            embeddings_dir,
            nome_saida
        )


        np.save(
            caminho_saida,
            embedding
        )


        sucesso_pessoa += 1

        total_sucesso += 1


        print(
            "   ✅ Embedding criado."
        )


    # ========================================================
    # RESULTADO DA PESSOA
    # ========================================================

    relatorio[pessoa] = {
        "total": len(fotos),
        "sucesso": sucesso_pessoa,
        "sem_rosto": sem_rosto_pessoa,
        "multiplos": multiplos_pessoa,
        "erro_imagem": erro_imagem_pessoa,
        "embedding_invalido":
            embedding_invalido_pessoa
    }


    print()

    print(
        f"Resultado {pessoa.upper()}: "
        f"{sucesso_pessoa}/{len(fotos)}"
    )

    print()


# ============================================================
# RESULTADO FINAL
# ============================================================

print()
print("=" * 70)
print("                  RESULTADO FINAL")
print("=" * 70)
print()


# ============================================================
# RELATÓRIO POR PESSOA
# ============================================================

for pessoa, dados in relatorio.items():

    print(
        f"{pessoa.upper():12s} "
        f"{dados['sucesso']:2d}/"
        f"{dados['total']:2d} "
        f"embeddings"
    )


print()


# ============================================================
# ESTATÍSTICAS GERAIS
# ============================================================

print(
    f"Total de fotos: "
    f"{total_fotos}"
)

print(
    f"Embeddings criados: "
    f"{total_sucesso}"
)

print(
    f"Fotos sem rosto: "
    f"{total_sem_rosto}"
)

print(
    f"Fotos com múltiplos rostos: "
    f"{total_multiplos_rostos}"
)

print(
    f"Erros de imagem/InsightFace: "
    f"{total_erro_imagem}"
)

print(
    f"Embeddings inválidos: "
    f"{total_embedding_invalido}"
)

print()


# ============================================================
# TAXA DE SUCESSO
# ============================================================

if total_fotos > 0:

    taxa_sucesso = (
        total_sucesso
        /
        total_fotos
    ) * 100

else:

    taxa_sucesso = 0


print(
    f"Taxa de sucesso: "
    f"{taxa_sucesso:.2f}%"
)

print()


# ============================================================
# DIAGNÓSTICO
# ============================================================

print("=" * 70)
print("                    DIAGNÓSTICO")
print("=" * 70)
print()


if (
    total_sucesso
    ==
    total_fotos
):

    print(
        "🟢 EXCELENTE: todas as fotos "
        "geraram embeddings."
    )


elif taxa_sucesso >= 90:

    print(
        "🟢 Muito bom: mais de 90% "
        "das fotos foram aproveitadas."
    )


elif taxa_sucesso >= 75:

    print(
        "🟡 Atenção: algumas fotos "
        "precisam ser investigadas."
    )


else:

    print(
        "🔴 Problema: muitas fotos "
        "não geraram embeddings."
    )


print()


# ============================================================
# VERIFICAR QUANTIDADE POR PESSOA
# ============================================================

print(
    "VERIFICAÇÃO POR PESSOA:"
)

print()


problemas_pessoas = []


for pessoa, dados in relatorio.items():

    if dados["sucesso"] < 5:

        problemas_pessoas.append(
            pessoa
        )

        print(
            f"🔴 {pessoa}: "
            f"poucas amostras válidas "
            f"({dados['sucesso']})"
        )

    elif dados["sucesso"] < dados["total"]:

        print(
            f"🟡 {pessoa}: "
            f"{dados['sucesso']}/"
            f"{dados['total']} válidas"
        )

    else:

        print(
            f"🟢 {pessoa}: "
            f"{dados['sucesso']}/"
            f"{dados['total']} válidas"
        )


print()


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("             BUILD EMBEDDINGS FINALIZADO")
print("=" * 70)
print()

print(
    "Os novos embeddings foram salvos em:"
)

print(
    "faces/<pessoa>/embeddings/"
)

print()

print(
    "O projeto antigo NÃO foi alterado."
)

print()