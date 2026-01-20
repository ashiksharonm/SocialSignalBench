import cv2
import numpy as np
import os


class FacePreprocessor:
    def __init__(self, config):
        self.target_size = tuple(
            config["preprocessing"]["face"]["image_size"]
        )  # (128, 128)
        self.detect_face = config["preprocessing"]["face"]["detect_face"]

        # Load Haar Cascade
        # We try to load from cv2 data path first
        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        haar_model = os.path.join(
            cv2_base_dir, "data/haarcascade_frontalface_default.xml"
        )

        if not os.path.exists(haar_model):
            # Fallback for some installs
            haar_model = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(haar_model)

    def process_image(self, file_path):
        """
        Load image, detect face, crop, resize, normalize.
        Output shape: (H, W, 1) - Grayscale
        """
        try:
            img = cv2.imread(file_path)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.detect_face:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    # Take largest face
                    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                    x, y, w, h = faces[0]
                    face_img = gray[y : y + h, x : x + w]
                else:
                    # No face detected, use whole image (fallback)
                    face_img = gray
            else:
                face_img = gray

            # Resize
            resized = cv2.resize(face_img, self.target_size)

            # Normalize [0, 1]
            normalized = resized.astype(np.float32) / 255.0

            # Add channel dim
            normalized = np.expand_dims(normalized, axis=-1)

            return normalized

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def save_features(self, features, output_path):
        # Save as compressed numpy
        np.savez_compressed(output_path, features=features)
