import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


class CNNPipeline:
    def __init__(self, input_size=128, batch_size=32, epochs=20):
        self.input_size = input_size
        self.batch_size = batch_size
        self.epochs = epochs

    def prepare_data(self, train_path, val_path, test_path):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(self.input_size, self.input_size),
            batch_size=self.batch_size,
            class_mode='binary'
        )
        val_generator = val_datagen.flow_from_directory(
            val_path,
            target_size=(self.input_size, self.input_size),
            batch_size=self.batch_size,
            class_mode='binary'
        )
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(self.input_size, self.input_size),
            batch_size=1,
            class_mode='binary',
            shuffle=False
        )
        return train_generator, val_generator, test_generator

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.input_size, self.input_size, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, train_generator, val_generator, model_path):
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, save_best_only=True)
        ]
        history = model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        return history

    def evaluate_model(self, model, test_generator):
        y_pred = model.predict(test_generator)
        y_pred = (y_pred > 0.5).astype(int)
        y_true = test_generator.classes

        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"Confusion Matrix:\n{cm}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def run(self, train_path, val_path, test_path, model_path):
        train_generator, val_generator, test_generator = self.prepare_data(train_path, val_path, test_path)
        model = self.build_model()
        history = self.train_model(model, train_generator, val_generator, model_path)
        results = self.evaluate_model(model, test_generator)
        return history, results