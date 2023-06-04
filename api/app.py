import cv2
import numpy as np
import base64
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

trainer_path = 'trainer.yml'
if not os.path.exists(trainer_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)


def read_base64_image(base64_image):
    image_data = base64.b64decode(base64_image)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image


def train_model(images, id):
    faces = []
    labels = []
    for i, image in enumerate(images):
        img = read_base64_image(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5)
        for j, (x, y, w, h) in enumerate(detected_faces):
            face_img = gray[y:y+h, x:x+w]
            faces.append(face_img)
            labels.append(id)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if not os.path.exists(trainer_path):
        recognizer.train(faces, np.array(labels))
        recognizer.write(trainer_path)
    else:
        recognizer.update(faces, np.array(labels))
        recognizer.save(trainer_path)


@app.route('/process-images', methods=['POST'])
def process_images():
    images = request.json['images']
    id = int(request.json['userid'])

    train_model(images, id)
    return jsonify({'status': True})


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)
font = cv2.FONT_HERSHEY_SIMPLEX


@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    if 'image' not in data:
        return {'id': 'No image data provided.'}, 400

    image_data = data['image']
    try:
        image_data = base64.b64decode(image_data)
        img_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 7)
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                id, conf = recognizer.predict(roi_gray)
                if conf < 50:
                    return {'id': id,'message':"success"}
                else:
                    return {'id': "Not in database"}
        else:
            return {'id': 0000}
    except (ValueError, cv2.error) as e:
        return {'id': 'Invalid image data provided.'}, 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
