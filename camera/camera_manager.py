# ============================================================
# VISIONAI V2
# GERENCIADOR DE CÂMERA
# ============================================================

import cv2
import threading
import time


class CameraManager:

    def __init__(
        self,
        camera_id,
        width,
        height,
        buffer_size=1
    ):

        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.buffer_size = buffer_size

        self.camera = None

        # ====================================================
        # FRAME MAIS RECENTE
        # ====================================================

        self.latest_frame = None
        self.latest_frame_time = 0.0

        # ====================================================
        # MÉTRICAS DA CÂMERA
        # ====================================================

        self.frames_captured = 0

        # ====================================================
        # LOCK
        # ====================================================

        self.lock = threading.Lock()

        # ====================================================
        # CONTROLE DA THREAD
        # ====================================================

        self.running = False
        self.thread = None


    # ========================================================
    # ABRIR CÂMERA
    # ========================================================

    def open(self):

        print(
            f"[CAMERA] Abrindo câmera ID "
            f"{self.camera_id}..."
        )


        self.camera = cv2.VideoCapture(
            self.camera_id
        )


        if not self.camera.isOpened():

            raise RuntimeError(
                f"Não foi possível abrir "
                f"a câmera {self.camera_id}."
            )


        # ----------------------------------------------------
        # CONFIGURAR RESOLUÇÃO
        # ----------------------------------------------------

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height
        )


        # ----------------------------------------------------
        # REDUZIR BUFFER
        # ----------------------------------------------------

        self.camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            self.buffer_size
        )


        # ----------------------------------------------------
        # DESCOBRIR RESOLUÇÃO REAL
        # ----------------------------------------------------

        real_width = int(
            self.camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        real_height = int(
            self.camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )


        print(
            "[CAMERA] Câmera aberta."
        )

        print(
            f"[CAMERA] Resolução real: "
            f"{real_width}x{real_height}"
        )


    # ========================================================
    # INICIAR THREAD
    # ========================================================

    def start(self):

        if self.camera is None:

            raise RuntimeError(
                "A câmera precisa ser aberta "
                "antes de iniciar."
            )


        if self.running:

            return


        self.running = True


        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )


        self.thread.start()


        print(
            "[CAMERA] Thread de captura iniciada."
        )


    # ========================================================
    # LOOP DE CAPTURA
    # ========================================================

    def _capture_loop(self):

        while self.running:

            success, frame = (
                self.camera.read()
            )


            # ------------------------------------------------
            # FALHA NA CAPTURA
            # ------------------------------------------------

            if not success:

                time.sleep(0.01)

                continue


            # ------------------------------------------------
            # GUARDAR SOMENTE O FRAME MAIS RECENTE
            # ------------------------------------------------

            with self.lock:

                self.latest_frame = frame

                self.latest_frame_time = (
                    time.perf_counter()
                )

                # Contabilizar todos os frames
                # realmente recebidos da câmera.
                self.frames_captured += 1


        print(
            "[CAMERA] Loop de captura encerrado."
        )


    # ========================================================
    # OBTER FRAME MAIS RECENTE
    # ========================================================

    def get_latest_frame(self):

        with self.lock:

            if self.latest_frame is None:

                return None, None


            # ------------------------------------------------
            # COPY
            # ------------------------------------------------

            frame = (
                self.latest_frame.copy()
            )


            timestamp = (
                self.latest_frame_time
            )


        return frame, timestamp


    # ========================================================
    # OBTER MÉTRICAS
    # ========================================================

    def get_metrics(self):

        with self.lock:

            return {
                "frames_captured":
                    self.frames_captured
            }


    # ========================================================
    # PARAR CÂMERA
    # ========================================================

    def stop(self):

        if not self.running:

            return


        print(
            "[CAMERA] Encerrando câmera..."
        )


        self.running = False


        # ----------------------------------------------------
        # AGUARDAR THREAD
        # ----------------------------------------------------

        if self.thread is not None:

            self.thread.join(
                timeout=2.0
            )


        # ----------------------------------------------------
        # LIBERAR CÂMERA
        # ----------------------------------------------------

        if self.camera is not None:

            self.camera.release()

            self.camera = None


        print(
            "[CAMERA] Câmera liberada."
        )