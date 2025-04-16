from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import pymongo
from datetime import datetime

app = Flask(__name__)
camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# MongoDB Setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['attendance_db']
collection = db['students']

# Create dataset folder
if not os.path.exists('static/dataset'):
    os.makedirs('static/dataset')


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():  # For video streaming
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    data = request.json
    name = data['name']
    student_id = data['id']
    class_name = data['class']
    year = data['year']
    mentor_no = data['mentor_no']
    mentor_name = data['mentor_name']

    # Save to MongoDB
    existing = collection.find_one({'id': student_id})
    if not existing:
        collection.insert_one({
            'name': name,
            'id': student_id,
            'class': class_name,
            'year': year,
            'mentor_no': mentor_no,
            'mentor_name': mentor_name,
            'created_at': datetime.now()
        })

    # Create student-specific directory
    student_dir = f"static/dataset/{name}_{student_id}/"
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)

    # Start capturing 20 face images
    count = 0
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            filename = f"{student_dir}{count}.jpg"
            if cv2.imwrite(filename, face_img):
                print(f"Saved {filename}")
            else:
                print(f"Failed to save {filename}")

        if count >= 60:
            break

    return jsonify({'status': 'success', 'message': f'{count} face images saved'})


if __name__ == '__main__':
    app.run(debug=True)
