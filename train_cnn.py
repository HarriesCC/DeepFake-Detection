import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0

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


# Data generators
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1/255)
    test_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        directory=train_path,
        target_size=(input_size, input_size),
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size_num,
        shuffle=True
    )
    val_generator = validation_datagen.flow_from_directory(
        directory=validation_path,
        target_size=(input_size, input_size),
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size_num,
        shuffle=True
    )
    test_generator = test_datagen.flow_from_directory(
        directory=test_path,
        classes=['real', 'fake'],
        target_size=(input_size, input_size),
        color_mode="rgb",
        class_mode=None,
        batch_size=1,
        shuffle=False
    )

    return train_generator, val_generator, test_generator


train_generator, validation_generator, test_generator = create_data_generators()


# Build model
def build_model(input_size):
    efficient_net = EfficientNetB0(
        weights='imagenet',
        input_shape=(input_size, input_size, 3),
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


model = build_model(input_size)


# Compile model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
custom_callbacks = [
    EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1),
    ModelCheckpoint(filepath=os.path.join(models_filepath, 'deepfake_model.h5'), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
]

# Train model
history = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=custom_callbacks
)
print(history.history)

# Load the best model
best_model = load_model(os.path.join(models_filepath, 'deepfake_model.h5'))

# Generate predictions
test_generator.reset()
preds = best_model.predict(test_generator, verbose=1)

# Save predictions to DataFrame
test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})

print(test_results)
