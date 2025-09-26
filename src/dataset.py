# %%
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# For images
def load_image_dataset(folder, labels_map, target_size=(224,224)):
    X, y = [], []
    for label_name, label_idx in labels_map.items():
        path = os.path.join(folder, label_name)
        if not os.path.exists(path):
            continue
        for fname in os.listdir(path):
            img = cv2.imread(os.path.join(path, fname))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            X.append(img / 255.0)
            y.append(label_idx)
    X = np.array(X)
    y = to_categorical(y, num_classes=len(labels_map))
    return X, y

# For short videos -> frames
def extract_frames_from_video(video_path, timesteps=16, target_size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < timesteps:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame/255.0)
    cap.release()
    # pad or trim
    if len(frames) < timesteps:
        while len(frames) < timesteps:
            frames.append(frames[-1])
    return np.array(frames[:timesteps])


# %%



