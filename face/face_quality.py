import cv2
import numpy as np


# ============================================================
# VISIONAI V2
# FACE QUALITY GATE
# ============================================================


class FaceQuality:

    def __init__(
        self,
        min_face_size=45,
        min_detection_score=0.45,
        min_brightness=35,
        max_brightness=230,
        min_sharpness=40
    ):

        self.min_face_size = min_face_size
        self.min_detection_score = min_detection_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_sharpness = min_sharpness


    # ========================================================
    # TAMANHO DO ROSTO
    # ========================================================

    def check_size(self, face):

        x1, y1, x2, y2 = (
            np.asarray(
                face.bbox,
                dtype=np.float32
            ).astype(int)
        )

        width = x2 - x1
        height = y2 - y1

        valid = (
            width >= self.min_face_size
            and
            height >= self.min_face_size
        )

        return valid, width, height


    # ========================================================
    # SCORE DE DETECÇÃO
    # ========================================================

    def check_detection_score(self, face):

        score = float(
            face.det_score
        )

        return (
            score >= self.min_detection_score,
            score
        )


    # ========================================================
    # BRILHO
    # ========================================================

    def check_brightness(
        self,
        frame,
        face
    ):

        x1, y1, x2, y2 = (
            np.asarray(
                face.bbox,
                dtype=np.float32
            ).astype(int)
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(
            frame.shape[1],
            x2
        )

        y2 = min(
            frame.shape[0],
            y2
        )


        if x2 <= x1 or y2 <= y1:

            return False, 0.0


        face_crop = frame[
            y1:y2,
            x1:x2
        ]


        if face_crop.size == 0:

            return False, 0.0


        gray = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY
        )


        brightness = float(
            np.mean(gray)
        )


        valid = (
            self.min_brightness
            <=
            brightness
            <=
            self.max_brightness
        )


        return valid, brightness


    # ========================================================
    # NITIDEZ
    # ========================================================

    def check_sharpness(
        self,
        frame,
        face
    ):

        x1, y1, x2, y2 = (
            np.asarray(
                face.bbox,
                dtype=np.float32
            ).astype(int)
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(
            frame.shape[1],
            x2
        )

        y2 = min(
            frame.shape[0],
            y2
        )


        if x2 <= x1 or y2 <= y1:

            return False, 0.0


        face_crop = frame[
            y1:y2,
            x1:x2
        ]


        if face_crop.size == 0:

            return False, 0.0


        gray = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY
        )


        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )


        valid = (
            sharpness
            >=
            self.min_sharpness
        )


        return valid, sharpness


    # ========================================================
    # AVALIAÇÃO COMPLETA
    # ========================================================

    def evaluate(
        self,
        frame,
        face
    ):

        size_ok, width, height = (
            self.check_size(
                face
            )
        )


        detection_ok, detection_score = (
            self.check_detection_score(
                face
            )
        )


        brightness_ok, brightness = (
            self.check_brightness(
                frame,
                face
            )
        )


        sharpness_ok, sharpness = (
            self.check_sharpness(
                frame,
                face
            )
        )


        quality_ok = (
            size_ok
            and
            detection_ok
            and
            brightness_ok
            and
            sharpness_ok
        )


        return {

            "quality_ok":
                quality_ok,

            "size_ok":
                size_ok,

            "width":
                width,

            "height":
                height,

            "detection_ok":
                detection_ok,

            "detection_score":
                detection_score,

            "brightness_ok":
                brightness_ok,

            "brightness":
                brightness,

            "sharpness_ok":
                sharpness_ok,

            "sharpness":
                sharpness
        }