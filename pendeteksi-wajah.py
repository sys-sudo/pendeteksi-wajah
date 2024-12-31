import subprocess
import os
import sys

def install_and_upgrade(package):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'show', package])
    except subprocess.CalledProcessError:
        print(f"Package '{package}' not found. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--break-system-packages'])
    else:
        print(f"Package '{package}' is already installed. Upgrading...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package, '--break-system-packages'])

install_and_upgrade('opencv-python')
install_and_upgrade('numpy')

import cv2
import os
import numpy as np

models_folder = 'models'

MAX_WIDTH = 700
MAX_HEIGHT = 600

def check_and_load_models():
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    prototxt_files = [f for f in os.listdir(models_folder) if f.endswith('.prototxt')]
    caffemodel_files = [f for f in os.listdir(models_folder) if f.endswith('.caffemodel')]

    models = []
    for prototxt in prototxt_files:
        caffemodel = None
        if 'gender' in prototxt:
            caffemodel = 'gender_net.caffemodel'
        elif 'age' in prototxt:
            caffemodel = 'age_net.caffemodel'

        if caffemodel and caffemodel in caffemodel_files:
            prototxt_path = os.path.join(models_folder, prototxt)
            caffemodel_path = os.path.join(models_folder, caffemodel)
            try:
                net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                models.append((prototxt_path, caffemodel_path, net))
            except Exception as e:
                print(f"Failed to load model {caffemodel}: {e}")

    if not models:
        print(f"Warning: No models found in '{models_folder}' folder.")
    return models


def load_haar_cascades():
    cascade_files = {
        'face': 'haarcascade_frontalface_default.xml',
        'eye': 'haarcascade_eye.xml',
        'eye_glasses': 'haarcascade_eye_tree_eyeglasses.xml',
        'smile': 'haarcascade_smile.xml',
        'nose': 'haarcascade_mcs_nose.xml',
        'fullbody': 'haarcascade_fullbody.xml',
        'upperbody': 'haarcascade_upperbody.xml',
        'lowerbody': 'haarcascade_lowerbody.xml',
        'hands': 'haarcascade_hand.xml'
    }
    cascades = {}
    for feature, filename in cascade_files.items():
        path = os.path.join(models_folder, filename)
        try:
            if os.path.exists(path):
                cascades[feature] = cv2.CascadeClassifier(path)
            else:
                print(f"Warning: Cascade file for {feature} not found.")
        except Exception as e:
            print(f"Error loading cascade {feature}: {e}")
    return cascades


gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def resize_frame(frame, max_width, max_height):
    height, width = frame.shape[:2]

    if width > max_width or height > max_height:
        aspect_ratio = width / height

        if width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        elif height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

        frame = cv2.resize(frame, (new_width, new_height))
    
    return frame

def load_human_detection_model():
    net = cv2.dnn.readNetFromCaffe(
        os.path.join(models_folder, 'deploy.prototxt'),
        os.path.join(models_folder, 'mobilenet_iter_73000.caffemodel')
    )
    return net

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def detect_and_display(models, cascades, human_detection_net):
    available_cameras = []
    for i in range(5): 
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    if not available_cameras:
        print("No cameras found.")
        return

    print("Available cameras:")
    for idx, cam_id in enumerate(available_cameras):
        print(f"{idx+1}. Camera {cam_id}")

    current_camera = available_cameras[0]
    cap = cv2.VideoCapture(current_camera)

    if not cap.isOpened():
        print(f"Error: Unable to access camera {current_camera}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    raw_mode = False
    auto_tracking = False  
    tracking_smoothness = 0.11
    center_x, center_y = 50, 50  
    prev_faces = None  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = resize_frame(frame, MAX_WIDTH, MAX_HEIGHT)

        if not raw_mode:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
            human_detection_net.setInput(blob)
            detections = human_detection_net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  
                    left = int(detections[0, 0, i, 3] * frame.shape[1])
                    top = int(detections[0, 0, i, 4] * frame.shape[0])
                    right = int(detections[0, 0, i, 5] * frame.shape[1])
                    bottom = int(detections[0, 0, i, 6] * frame.shape[0])

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if 'face' in cascades:
                faces = cascades['face'].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30))
                if len(faces) > 0:
                    faces = non_max_suppression(np.array(faces))  
                    prev_faces = faces 
                elif prev_faces is not None:
                    faces = prev_faces 

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]

                    face_center_x = x + w // 2
                    face_center_y = y + h // 2

                    if auto_tracking:  
                        center_x = int(center_x + tracking_smoothness * (face_center_x - center_x))
                        center_y = int(center_y + tracking_smoothness * (face_center_y - center_y))

                        roi_width = 558
                        roi_height = 489
                        x_offset = max(0, min(center_x - roi_width // 2, frame.shape[1] - roi_width))
                        y_offset = max(0, min(center_y - roi_height // 2, frame.shape[0] - roi_height))

                        frame = frame[y_offset:y_offset + roi_height, x_offset:x_offset + roi_width]

                    if 'eye' in cascades:
                        eyes = cascades['eye'].detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7, minSize=(20, 20))
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    if 'eye_glasses' in cascades:
                        eyes_glasses = cascades['eye_glasses'].detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7, minSize=(20, 20))
                        for (ex, ey, ew, eh) in eyes_glasses:
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

                    if 'smile' in cascades:
                        smiles = cascades['smile'].detectMultiScale(
                            roi_gray,
                            scaleFactor=1.9,
                            minNeighbors=25,
                            minSize=(25, 25)
                        )
                        for (sx, sy, sw, sh) in smiles:
                            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                            cv2.putText(frame, "Smile", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if 'nose' in cascades:
                        noses = cascades['nose'].detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
                        for (nx, ny, nw, nh) in noses:
                            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

                    if models:
                        face_blob = cv2.dnn.blobFromImage(frame[y:y+h, x:x+w], 1.0, (227, 227), (78.42, 87.76, 114.93), swapRB=False)
                        for prototxt_path, caffemodel_path, net in models:
                            try:
                                net.setInput(face_blob)
                                preds = net.forward()

                                confidence_threshold = 0.6

                                if 'gender' in prototxt_path:
                                    gender_idx = preds[0].argmax()
                                    confidence = preds[0][gender_idx]
                                    if confidence > confidence_threshold:
                                        gender = gender_list[gender_idx]
                                        cv2.putText(frame, f"Gender: {gender} ({confidence*100:.2f}%)", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                elif 'age' in prototxt_path:
                                    age_idx = preds[0].argmax()
                                    confidence = preds[0][age_idx]
                                    if confidence > confidence_threshold:
                                        age = age_list[age_idx]
                                        cv2.putText(frame, f"Age: {age} ({confidence*100:.2f}%)", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            except Exception as e:
                                print(f"Error during model inference: {e}")

        cv2.imshow('Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            raw_mode = not raw_mode
        elif key == ord('t'):  
            auto_tracking = not auto_tracking
        elif key == ord('1') and len(available_cameras) > 1: 
            current_camera = available_cameras[(available_cameras.index(current_camera) + 1) % len(available_cameras)]
            cap.release()
            cap = cv2.VideoCapture(current_camera)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        models = check_and_load_models()
        cascades = load_haar_cascades()
        human_detection_net = load_human_detection_model() 
        detect_and_display(models, cascades, human_detection_net)
    except Exception as e:
        print(f"Error: {e}")
