import cv2
from mtcnn import MTCNN
import os
import json
import tensorflow as tf
from keras import backend as K


def configure_tensorflow():
    print(tf.__version__)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only


def get_all_directories(directory):
    files_and_folders = os.listdir(directory)
    dirs = [f for f in files_and_folders if os.path.isdir(os.path.join(directory, f))]
    return dirs


def process_images(base_path, is_fake):
    # with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    #     metadata = json.load(metadata_json)
    #     print(f"Total files in metadata: {len(metadata)}")

    dirs = get_all_directories(base_path)

    output_path = './train_dataset/train/fake' if is_fake else './train_dataset/train/real'

    for filename in dirs:
        tmp_path = os.path.join(base_path, get_filename_only(filename))
        print(f'Processing Directory: {tmp_path}')
        frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
        faces_path = os.path.join(tmp_path, 'faces')
        print(f'Creating Directory: {faces_path}')
        os.makedirs(faces_path, exist_ok=True)
        print('Cropping Faces from Images...')

        for frame in frame_images:

            #if not png file, skip
            if not frame.endswith('.png'):
                continue
            print(f'Processing {frame}')
            detector = MTCNN()
            image = cv2.cvtColor(cv2.imread(os.path.join(tmp_path, frame)), cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(image)
            print(f'Face Detected: {len(results)}')
            count = 0

            for result in results:
                bounding_box = result['box']
                print(bounding_box)
                confidence = result['confidence']
                print(confidence)
                if len(results) < 2 or confidence > 0.95:
                    margin_x = bounding_box[2] * 0.3  # 30% as the margin
                    margin_y = bounding_box[3] * 0.3  # 30% as the margin
                    x1 = max(int(bounding_box[0] - margin_x), 0)
                    x2 = min(int(bounding_box[0] + bounding_box[2] + margin_x), image.shape[1])
                    y1 = max(int(bounding_box[1] - margin_y), 0)
                    y2 = min(int(bounding_box[1] + bounding_box[3] + margin_y), image.shape[0])
                    print(x1, y1, x2, y2)
                    crop_image = image[y1:y2, x1:x2]
                    new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, get_filename_only(frame)), count)
                    count += 1
                    cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
                    output_filename = '{}-{:02d}.png'.format(os.path.join(output_path, get_filename_only(frame)), count)
                    cv2.imwrite(output_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
                else:
                    print('Skipped a face..')


# Example usage
if __name__ == "__main__":
    # base_path = './raw_dataset/manipulated_sequences/DeepFakeDetection/c40/videos/output/'
    base_path = './raw_dataset/original_sequences/actors/c40/videos/output/'
    configure_tensorflow()
    process_images(base_path, is_fake=False)
