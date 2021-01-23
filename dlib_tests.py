import dlib
from pathlib import Path
import cv2
import numpy as np
from itertools import chain

THRESHOLD = 70

detector = None
sp = None
facerec = None
ref = None


def compute_ref():
    global ref
    img = cv2.imread("Photos/2021-01-21-181917.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    det = next(iter(dets))
    shape = sp(gray, det)
    ref = np.array(facerec.compute_face_descriptor(img, shape))


def setup():
    global detector, sp, facerec
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("dlib_models/shape_predictor_5_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_models/dlib_face_recognition_resnet_model_v1.dat")

    compute_ref()


def process(img):
    dets = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
    for det in dets:
        p1 = (det.left(), det.top())
        p2 = (det.right(), det.bottom())

        shape = sp(img, det)
        face_descriptor = np.array(facerec.compute_face_descriptor(img, shape))
        dist = np.sum((face_descriptor - ref) ** 2)
        percentage = 100 * (1 - 2 / np.pi * np.arctan(dist))

        color = (0, 0, 255) if percentage < THRESHOLD else (0, 255, 0)

        img = cv2.rectangle(img, p1, p2, color, 2)

        cv2.putText(img, str(round(percentage)), p2, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return img


def run_live():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        res = process(frame)
        cv2.imshow('webcam', res)

        if cv2.waitKey(10) & 0xFF in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()
    print("Done!")


def run_on_folder(path: Path):
    images_paths = chain(path.glob("*.jpg"), path.glob("*.png"))
    for img_path in images_paths:
        print(img_path)

        img = cv2.imread(img_path.as_posix())
        if img.shape[1] > 1000:
            fx = 640 / img.shape[1]
            img = cv2.resize(img, None, fx=fx, fy=fx)
            print(img.shape)
        res = process(img)
        cv2.imwrite("Results/" + img_path.name, res)


if __name__ == "__main__":
    setup()
    # run_on_folder(Path("Photos"))
    run_live()
