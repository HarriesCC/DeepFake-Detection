import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.utils import custom_object_scope

print('TensorFlow version: ', tf.__version__)

# Set paths
dataset_path = './train_dataset/'
tmp_debug_path = './tmp_debug'
models_filepath = './models'

# Create directories if they don't exist
for path in [tmp_debug_path, models_filepath]:
    print(f'Creating Directory: {path}')
    os.makedirs(path, exist_ok=True)


def get_filename_only(file_path):
    return os.path.basename(file_path).split('.')[0]


# Set parameters
input_size = 128
batch_size_num = 32
num_epochs = 20
train_path = os.path.join(dataset_path, 'train')
validation_path = os.path.join(dataset_path, 'validation')
test_path = os.path.join(dataset_path, 'test')


class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape, seed, **kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return super().call(inputs)


# 加载训练好的模型
with custom_object_scope({'FixedDropout': FixedDropout}):
    best_model = load_model(os.path.join(models_filepath, 'deepfake_model.h5'))


def load_and_preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict_image(url):
    img_array = load_and_preprocess_image(url, (input_size, input_size))
    prediction = best_model.predict(img_array)
    prediction_label = 'real' if prediction[0][0] >= 0.5 else 'fake'

    print(prediction[0][0])
    print(f"The image is predicted to be: {prediction_label}")
    return prediction_label


predict_image("train_dataset/test/fake/01_11__talking_against_wall__9229VVZ3-000-01.png")