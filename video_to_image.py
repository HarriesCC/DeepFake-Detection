import json
import os
import cv2
import math


def process_videos(base_path, output_path):
    def get_filename_only(file_path):
        file_basename = os.path.basename(file_path)
        filename_only = file_basename.split('.')[0]
        return filename_only

    def get_all_files(directory):
        files_and_folders = os.listdir(directory)
        files = [f for f in files_and_folders if os.path.isfile(os.path.join(directory, f))]
        return files

    # with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    #     metadata = json.load(metadata_json)
    #     print(f"Total files in metadata: {len(metadata)}")

    files = get_all_files(base_path)
    for filename in files:
        print(f"Processing file: {filename}")
        if filename.endswith(".mp4"):
            if output_path:
                tmp_path = os.path.join(output_path, get_filename_only(filename))
            else:
                tmp_path = os.path.join(base_path, get_filename_only(filename))
            print(f'Creating Directory: {tmp_path}')
            os.makedirs(tmp_path, exist_ok=True)
            print('Converting Video to Images...')
            count = 0
            video_file = os.path.join(base_path, filename)
            cap = cv2.VideoCapture(video_file)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            while cap.isOpened():
                frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % math.floor(frame_rate) == 0:
                    print('Original Dimensions: ', frame.shape)
                    if frame.shape[1] < 300:
                        scale_ratio = 2
                    elif frame.shape[1] > 1900:
                        scale_ratio = 0.33
                    elif 1000 < frame.shape[1] <= 1900:
                        scale_ratio = 0.5
                    else:
                        scale_ratio = 1
                    print('Scale Ratio: ', scale_ratio)

                    width = int(frame.shape[1] * scale_ratio)
                    height = int(frame.shape[0] * scale_ratio)
                    dim = (width, height)
                    new_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                    print('Resized Dimensions: ', new_frame.shape)

                    new_filename = '{}-{:03d}.png'.format(os.path.join(tmp_path, get_filename_only(filename)), count)
                    count += 1
                    cv2.imwrite(new_filename, new_frame)
            cap.release()
            print("Done!")
        else:
            continue


if __name__ == '__main__':
    base_path = './raw_dataset/manipulated_sequences/DeepFakeDetection/c40/videos/'
    output_path = './raw_dataset/manipulated_sequences/DeepFakeDetection/c40/videos/output/'
    process_videos(base_path, output_path)