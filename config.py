# ============================================================
# VISIONAI V2 - CONFIGURAÇÕES
# ============================================================


# ------------------------------------------------------------
# MODELO
# ------------------------------------------------------------

MODEL_NAME = "buffalo_sc"


# ------------------------------------------------------------
# BANCO DE FACES
# ------------------------------------------------------------

FACES_DIR = "faces"


# ------------------------------------------------------------
# CÂMERA
# ------------------------------------------------------------

CAMERA_ID = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Mantém somente os frames mais recentes
# para reduzir atraso da câmera.

CAMERA_BUFFER_SIZE = 1


# ------------------------------------------------------------
# INSIGHTFACE
# ------------------------------------------------------------

DET_SIZE = (320, 320)


# ------------------------------------------------------------
# RECONHECIMENTO
# ------------------------------------------------------------

# Threshold atual utilizado pelo sistema.
#
# IMPORTANTE:
# Este valor ainda não é considerado definitivo.
# A calibração final será feita posteriormente
# com benchmark controlado.

RECOGNITION_THRESHOLD = 0.40