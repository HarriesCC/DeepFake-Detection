import os
import cv2
import math
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0
import random
import numpy as np
import pandas as pd
import shutil
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


class DeepfakeDetectionPipeline:
    def __init__(self, input_size=128, batch_size=32, epochs=20):
        self.input_size = input_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.detector = MTCNN()
        self.configure_tensorflow()

    def configure_tensorflow(self):
        print(tf.__version__)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # 在导入或使用任何TensorFlow功能之前设置内存增长
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")

    def process_videos(self, base_path, output_path):
        for category in ['real_videos', 'fake_videos']:
            category_path = os.path.join(base_path, category)
            if not os.path.exists(category_path):
                print(f"Warning: {category_path} does not exist. Skipping.")
                continue

            files = [f for f in os.listdir(category_path) if f.endswith(".mp4")]
            for filename in files:
                print(f"Processing {category} file: {filename}")
                video_file = os.path.join(category_path, filename)
                tmp_path = os.path.join(output_path, category.split('_')[0], os.path.splitext(filename)[0])
                os.makedirs(tmp_path, exist_ok=True)

                cap = cv2.VideoCapture(video_file)
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                count = 0
                while cap.isOpened():
                    frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_id % math.floor(frame_rate) == 0:
                        new_frame = self.resize_frame(frame)
                        new_filename = f'{tmp_path}/{count:03d}.png'
                        cv2.imwrite(new_filename, new_frame)
                        count += 1
                cap.release()
                print(f"Extracted {count} frames from {filename}")

    def resize_frame(self, frame):
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
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def crop_faces(self, input_path, output_path, is_fake):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.png'):
                    print(f"Processing file: {file}")
                    image_path = os.path.join(root, file)
                    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    results = self.detector.detect_faces(image)
                    for i, result in enumerate(results):
                        if len(results) < 2 or result['confidence'] > 0.95:
                            bounding_box = result['box']
                            margin_x = bounding_box[2] * 0.3
                            margin_y = bounding_box[3] * 0.3
                            x1 = max(int(bounding_box[0] - margin_x), 0)
                            x2 = min(int(bounding_box[0] + bounding_box[2] + margin_x), image.shape[1])
                            y1 = max(int(bounding_box[1] - margin_y), 0)
                            y2 = min(int(bounding_box[1] + bounding_box[3] + margin_y), image.shape[0])
                            crop_image = image[y1:y2, x1:x2]

                            parent_dir = os.path.basename(os.path.dirname(image_path))
                            sanitized_parent_dir = parent_dir.replace('/', '_')

                            output_filename = f'{output_path}/{"fake" if is_fake else "real"}/{sanitized_parent_dir + "_" + os.path.splitext(file)[0]}_{i}.png'
                            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                            cv2.imwrite(output_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))

    def prepare_dataset(self, train_path, val_path, test_path):
        train_datagen = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1 / 255)
        test_datagen = ImageDataGenerator(rescale=1 / 255)

        train_generator = train_datagen.flow_from_directory(
            directory=train_path,
            target_size=(self.input_size, self.input_size),
            color_mode="rgb",
            class_mode="binary",
            batch_size=self.batch_size,
            shuffle=True
        )
        val_generator = val_datagen.flow_from_directory(
            directory=val_path,
            target_size=(self.input_size, self.input_size),
            color_mode="rgb",
            class_mode="binary",
            batch_size=self.batch_size,
            shuffle=True
        )
        test_generator = test_datagen.flow_from_directory(
            directory=test_path,
            classes=['real', 'fake'],
            target_size=(self.input_size, self.input_size),
            color_mode="rgb",
            class_mode=None,
            batch_size=1,
            shuffle=False
        )

        return train_generator, val_generator, test_generator

    def build_model(self):
        efficient_net = EfficientNetB0(
            weights='imagenet',
            input_shape=(self.input_size, self.input_size, 3),
            include_top=False,
            pooling='max'
        )

        model = Sequential([
            efficient_net,
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.summary()
        return model

    def train_model(self, train_generator, val_generator, models_filepath):
        model = self.build_model()
        model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1),
            ModelCheckpoint(filepath=os.path.join(models_filepath, 'deepfake_model.h5'),
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        ]

        history = model.fit(
            train_generator,
            epochs=self.epochs,
            steps_per_epoch=len(train_generator),
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=callbacks
        )
        return history, model

    def split_dataset(self, source_dir, train_dir, val_dir, test_dir):
        for category in ['real', 'fake']:
            # Create target directories
            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(val_dir, category), exist_ok=True)
            os.makedirs(os.path.join(test_dir, category), exist_ok=True)

            # Get all image files
            all_images = [f for f in os.listdir(os.path.join(source_dir, category)) if f.endswith('.png')]

            # Group images by video
            video_groups = {}
            for img in all_images:
                video_name = '_'.join(img.split('_')[:-2])
                if video_name not in video_groups:
                    video_groups[video_name] = []
                video_groups[video_name].append(img)

            # Shuffle the video groups
            video_names = list(video_groups.keys())
            random.shuffle(video_names)

            # Calculate split points
            n_videos = len(video_names)
            train_split = int(0.8 * n_videos)
            val_split = int(0.9 * n_videos)

            # Split and move files
            for i, video_name in enumerate(video_names):
                if i < train_split:
                    target_dir = train_dir
                elif i < val_split:
                    target_dir = val_dir
                else:
                    target_dir = test_dir

                for img in video_groups[video_name]:
                    src = os.path.join(source_dir, category, img)
                    dst = os.path.join(target_dir, category, img)
                    shutil.copy(src, dst)

        print(f"Dataset split complete.")
        print(
            f"Train videos: {train_split}, Validation videos: {val_split - train_split}, Test videos: {n_videos - val_split}")
        print(
            f"Ratio - Train: {train_split / n_videos:.2f}, Validation: {(val_split - train_split) / n_videos:.2f}, Test: {(n_videos - val_split) / n_videos:.2f}")


    def evaluate_model(self, model, test_generator):
        # 预测测试集
        test_generator.reset()
        y_pred_proba = model.predict(test_generator, steps=len(test_generator))
        y_pred = (y_pred_proba > 0.5).astype(int)  # 使用 0.5 作为初始阈值

        # 获取真实标签
        y_true = test_generator.classes

        # 确保标签的顺序正确
        if not np.array_equal(test_generator.class_indices, {'real': 0, 'fake': 1}):
            y_true = 1 - y_true
            y_pred = 1 - y_pred
            y_pred_proba = 1 - y_pred_proba

        # 计算各种指标
        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred_proba)

        print("Confusion Matrix:")
        print(cm)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

        # 尝试找到最佳阈值（基于 F1 分数）
        thresholds = np.arange(0, 1, 0.01)
        f1_scores = [f1_score(y_true, (y_pred_proba > t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)

        print(f"Best threshold: {best_threshold:.2f}")
        print(f"Best F1 Score: {best_f1:.4f}")

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'best_threshold': best_threshold,
            'best_f1': best_f1
        }

    def run_pipeline(self, video_path, output_path, dataset_path, models_path):
        # Step 1: Process videos and extract frames
        # self.process_videos(video_path, output_path)
        #
        # # Step 2: Crop faces
        # self.crop_faces(output_path+ '/real', dataset_path , is_fake=False)  # For real videos
        # self.crop_faces(output_path + '/fake', dataset_path, is_fake=True)  # For real videos
        # Repeat for fake videos with is_fake=True

        train_dir = os.path.join(dataset_path, 'train')
        val_dir = os.path.join(dataset_path, 'validation')
        test_dir = os.path.join(dataset_path, 'test')

        # empty the directories
        for directory in [train_dir, val_dir, test_dir]:
            for category in ['real', 'fake']:
                shutil.rmtree(os.path.join(directory, category), ignore_errors=True)

        self.split_dataset(dataset_path, train_dir, val_dir, test_dir)

        # Step 3: Prepare dataset
        train_generator, val_generator, test_generator = self.prepare_dataset(
            train_dir, val_dir, test_dir
        )
        #
        # # Step 4: Train model
        # history, model = self.train_model(train_generator, val_generator, models_path)
        # print(history.history)

        # Step 5: Evaluate model
        # result = self.evaluate_model(model, test_generator)

        best_model = load_model(os.path.join(models_path, 'deepfake_model.h5'))

        test_generator.reset()

        preds = best_model.predict(
            test_generator,
            verbose=1
        )

        test_results = pd.DataFrame({
            "Filename": test_generator.filenames,
            "Prediction": preds.flatten()
        })
        print(test_results)

        # print(result)


if __name__ == "__main__":
    pipeline = DeepfakeDetectionPipeline()
    pipeline.run_pipeline(
        video_path='./raw_videos',
        output_path='./processed_frames',
        dataset_path='./dataset',
        models_path='./models'
    )