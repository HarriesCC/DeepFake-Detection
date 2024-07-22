from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 设置允许的文件扩展名
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

input_size = 128
models_filepath = './models'

class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape, seed, **kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return super().call(inputs)

# 加载模型
with custom_object_scope({'FixedDropout': FixedDropout}):
    best_model = load_model(os.path.join(models_filepath, 'deepfake_model.h5'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_frame(frame):
    # 调整帧大小
    if frame.shape[1] < 300:
        scale_ratio = 2
    elif frame.shape[1] > 1900:
        scale_ratio = 0.33
    elif 1000 < frame.shape[1] <= 1900:
        scale_ratio = 0.5
    else:
        scale_ratio = 1

    width = int(frame.shape[1] * scale_ratio)
    height = int(frame.shape[0] * scale_ratio)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # 预处理帧
    img = cv2.resize(resized_frame, (input_size, input_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

def predict_frame(frame):
    processed_frame = process_frame(frame)
    prediction = best_model.predict(processed_frame)
    prediction_label = 'real' if prediction[0][0] >= 0.5 else 'fake'
    confidence = prediction[0][0] if prediction_label == 'real' else 1 - prediction[0][0]
    return prediction_label, confidence

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, math.floor(frame_rate))  # 每秒采样一帧

    fake_frames = 0
    total_processed_frames = 0

    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % sample_interval == 0:
            prediction, _ = predict_frame(frame)
            if prediction == 'fake':
                fake_frames += 1
            total_processed_frames += 1

        # 打印处理进度
        if total_processed_frames % 10 == 0:
            print(f"Processed {total_processed_frames} frames out of {total_frames}")

    cap.release()

    fake_ratio = fake_frames / total_processed_frames if total_processed_frames > 0 else 0
    return 'fake' if fake_ratio > 0.5 else 'real', fake_ratio


@app.route('/predict', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            prediction, fake_ratio = process_video(file_path)
            return jsonify({
                'filename': filename,
                'prediction': prediction,
                'fake_ratio': fake_ratio
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # 删除上传的文件
            os.remove(file_path)
    else:
        return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    app.run(debug=True)