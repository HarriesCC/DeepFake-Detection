import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


class HybridPipeline:
    def __init__(self, input_size=128, batch_size=32, epochs=20, sequence_length=10):
        self.input_size = input_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length

    def prepare_data(self, train_path, val_path, test_path):
        def sequence_generator(generator, sequence_length):
            while True:
                batch_images = []
                batch_labels = []
                for _ in range(self.batch_size):
                    images = []
                    for _ in range(sequence_length):
                        img, label = next(generator)
                        images.append(img[0])
                    batch_images.append(np.array(images))
                    batch_labels.append(label[0])
                yield np.array(batch_images), np.array(batch_labels)

        train_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(self.input_size, self.input_size),
            batch_size=1,
            class_mode='binary'
        )
        val_generator = val_datagen.flow_from_directory(
            val_path,
            target_size=(self.input_size, self.input_size),
            batch_size=1,
            class_mode='binary'
        )
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(self.input_size, self.input_size),
            batch_size=1,
            class_mode='binary',
            shuffle=False
        )

        train_sequence = sequence_generator(train_generator, self.sequence_length)
        val_sequence = sequence_generator(val_generator, self.sequence_length)
        test_sequence = sequence_generator(test_generator, self.sequence_length)

        return train_sequence, val_sequence, test_sequence

    def build_model(self):
        input_layer = Input(shape=(self.sequence_length, self.input_size, self.input_size, 3))

        # CNN part
        cnn = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(input_layer)
        cnn = TimeDistributed(MaxPooling2D(2, 2))(cnn)
        cnn = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(cnn)
        cnn = TimeDistributed(MaxPooling2D(2, 2))(cnn)
        cnn = TimeDistributed(Flatten())(cnn)

        # RNN part
        rnn = LSTM(64, return_sequences=True)(cnn)
        rnn = LSTM(32)(rnn)

        # Output
        output = Dense(1, activation='sigmoid')(rnn)

        model = Model(inputs=input_layer, outputs=output)
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
        y_true = np.array([label for _, label in test_generator])

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


if __name__ == "__main__":
    train_path = "./dataset/train"
    val_path = "./dataset/validation"
    test_path = "./dataset/test"

