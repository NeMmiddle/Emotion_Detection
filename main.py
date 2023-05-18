import cv2
import numpy as np
from keras.models import load_model
from fps import calculate_fps, fps_coords


def detect_emotions():
    model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5", compile=False)
    emotion_labels = {
        0: "Злой",
        1: "Отвращение",
        2: "Страх",
        3: "Счастье",
        4: "Грусть",
        5: "Удивление",
        6: "Нейтральное",
    }

    cap = cv2.VideoCapture(0)

    start_time, fps, fps_text = calculate_fps()

    fps_x, fps_y, font, font_scale, font_thickness = fps_coords()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(gray)

        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

            roi = normalize_roi(roi_gray)

            preds = model.predict(roi)[0]
            label = emotion_labels[preds.argmax()]

            display_label(frame, label, x, y, w, h, font, font_scale, font_thickness)

        start_time, fps, fps_text = calculate_fps(start_time=start_time, fps=fps)

        if fps_text is not None:
            display_fps(frame, fps_text, fps_x, fps_y, font, font_scale, font_thickness)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_faces(gray):
    face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces


def normalize_roi(roi_gray):
    roi = roi_gray.astype("float") / 255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)
    return roi


def display_label(frame, label, x, y, w, h, font, font_scale, font_thickness):
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    cv2.putText(
        frame,
        label,
        (x, y - text_size[1] - 10),
        font,
        font_scale,
        (0, 0, 255),
        font_thickness,
    )
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


def display_fps(frame, fps_text, fps_x, fps_y, font, font_scale, font_thickness):
    if fps_text is not None:
        cv2.putText(
            frame,
            fps_text,
            (fps_x, fps_y),
            font,
            font_scale,
            (0, 0, 255),
            font_thickness,
        )


if __name__ == "__main__":
    detect_emotions()
