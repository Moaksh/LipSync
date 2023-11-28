import tensorflow as tf
from typing import List
import cv2
import os

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video1(path:str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])

    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def load_data1(path: str):
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
    frames = load_video1(video_path)
    return frames





import cv2
import dlib
import tensorflow as tf
import os
from typing import List


def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")


    resized_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            x_min, y_min = min(point[0] for point in lip_points)-20, min(point[1] for point in lip_points)-20
            x_max, y_max = max(point[0] for point in lip_points)+20, max(point[1] for point in lip_points)+20
            lip_crop = frame[y_min:y_max, x_min:x_max]

            resized_lip = cv2.resize(lip_crop, (140, 46))
            resized_lip_gray = tf.image.rgb_to_grayscale(resized_lip)

            # resized_frames.append(tf.cast(resized_lip_gray, tf.float32))
            resized_frames.append(resized_lip_gray)

    cap.release()

    if resized_frames:
        mean = tf.math.reduce_mean(resized_frames)
        std = tf.math.reduce_std(tf.cast(resized_frames, tf.float32))
        return tf.cast((resized_frames - mean), tf.float32) / std
        # frames_tensor = tf.stack(resized_frames)
        # mean = tf.math.reduce_mean(frames_tensor)
        # std = tf.math.reduce_std(frames_tensor)
        # return (frames_tensor - mean) / std
    else:
        return []
def load_data(path: str):
    if isinstance(path, bytes):
        path = path.decode()
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('..', 'data', 's1', f'{file_name}.mpg')
    frames = load_video(video_path)
    return frames

