# ============================================================
# VISIONAI V2
# PRIMEIRO TESTE DA ARQUITETURA
# ============================================================

import cv2
import time


# ============================================================
# CONFIGURAÇÕES
# ============================================================

from config import (
    CAMERA_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_BUFFER_SIZE,
    WINDOW_NAME,
    SHOW_DEBUG
)


# ============================================================
# CÂMERA
# ============================================================

from camera.camera_manager import (
    CameraManager
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("                    VISIONAI V2")
    print("=" * 60)
    print()


    # ========================================================
    # CRIAR GERENCIADOR DA CÂMERA
    # ========================================================

    camera = CameraManager(
        camera_id=CAMERA_ID,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        buffer_size=CAMERA_BUFFER_SIZE
    )


    # ========================================================
    # ABRIR CÂMERA
    # ========================================================

    try:

        camera.open()


    except RuntimeError as erro:

        print()
        print(
            f"[ERRO] {erro}"
        )
        print()

        return


    # ========================================================
    # INICIAR THREAD DA CÂMERA
    # ========================================================

    camera.start()


    print()
    print(
        "[SYSTEM] VisionAI V2 iniciado."
    )

    print(
        "[SYSTEM] Pressione Q para sair."
    )

    print()


    # ========================================================
    # CONTROLE DE FPS
    # ========================================================

    fps_counter = 0

    fps_timer = time.perf_counter()

    fps = 0


    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # PEGAR FRAME MAIS RECENTE
            # ------------------------------------------------

            frame, timestamp = (
                camera.get_latest_frame()
            )


            # ------------------------------------------------
            # AINDA NÃO TEM FRAME
            # ------------------------------------------------

            if frame is None:

                time.sleep(0.001)

                continue


            # ------------------------------------------------
            # CONTADOR DE FPS
            # ------------------------------------------------

            fps_counter += 1


            current_time = (
                time.perf_counter()
            )


            if (
                current_time
                - fps_timer
                >= 1.0
            ):

                fps = fps_counter

                fps_counter = 0

                fps_timer = current_time


            # =================================================
            # DEBUG
            # =================================================

            if SHOW_DEBUG:

                cv2.putText(
                    frame,
                    f"Camera FPS: {fps}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )


                cv2.putText(
                    frame,
                    "Camera: ONLINE",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )


            # =================================================
            # MOSTRAR FRAME
            # =================================================

            cv2.imshow(
                WINDOW_NAME,
                frame
            )


            # =================================================
            # TECLADO
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            # ------------------------------------------------
            # Q = SAIR
            # ------------------------------------------------

            if key == ord("q"):

                break


    finally:

        # ====================================================
        # ENCERRAMENTO SEGURO
        # ====================================================

        camera.stop()

        cv2.destroyAllWindows()


        print()
        print("=" * 60)
        print("             VISIONAI V2 ENCERRADO")
        print("=" * 60)
        print()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()