import cv2

from mtcnn import MTCNN

import sys, os.path

import json

from keras import backend as K

import tensorflow as tf

from logger import logging



os.chdir('..')

# PRINT TENSORFLOW VERSION
logging.info(f'TensorFlow version: {tf.__version__}')

print(tf.__version__)

# SET TENSORFLOW LOGGING LEVEL TO ERROR
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# LIST PHYSICAL GPU DEVICES
physical_devices = tf.config.list_physical_devices('GPU')

# PRINT PHYSICAL GPU DEVICES
logging.info(f'Physical GPU devices: {physical_devices}')

print(physical_devices)

# SET MEMORY GROWTH FOR THE FIRST GPU DEVICE
# tf.config.experimental.set_memory_growth(physical_devices, True)

base_path = 'train_sample_videos'


# FUNCTION TO GET FILENAME WITHOUT EXTENSION
def get_filename_only(file_path):

    file_basename = os.path.basename(file_path)

    filename_only = file_basename.split('.')[0]

    return filename_only


# LOAD METADATA JSON FILE
with open(os.path.join(base_path, 'metadata.json')) as metadata_json:

    metadata = json.load(metadata_json)

    # PRINT NUMBER OF METADATA ENTRIES
    logging.info(f'Number of metadata entries: {len(metadata)}')

    print(len(metadata))


# PROCESS EACH FILENAME IN METADATA
for filename in metadata.keys():

    # CREATE TEMPORARY PATH FOR THE VIDEO FILE
    tmp_path = os.path.join(base_path, get_filename_only(filename))

    # PRINT CURRENT DIRECTORY BEING PROCESSED
    logging.info(f'Processing Directory: {tmp_path}')

    print('Processing Directory: ' + tmp_path)

    # LIST ALL FRAME IMAGES IN THE TEMPORARY PATH
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]

    # CREATE A DIRECTORY FOR CROPPED FACES
    faces_path = os.path.join(tmp_path, 'faces')

    # PRINT THE FACES DIRECTORY BEING CREATED
    logging.info(f'Creating Directory: {faces_path}')

    print('Creating Directory: ' + faces_path)

    os.makedirs(faces_path, exist_ok=True)

    # PRINT MESSAGE ABOUT CROPPING FACES
    logging.info('Cropping Faces from Images...')

    print('Cropping Faces from Images...')


    # PROCESS EACH FRAME IMAGE
    for frame in frame_images:

        # PRINT CURRENT FRAME BEING PROCESSED
        logging.info(f'Processing frame: {frame}')

        print('Processing ', frame)

        # INITIALIZE MTCNN DETECTOR
        detector = MTCNN()

        # READ AND CONVERT IMAGE TO RGB
        image = cv2.cvtColor(cv2.imread(os.path.join(tmp_path, frame)), cv2.COLOR_BGR2RGB)

        # DETECT FACES IN THE IMAGE
        results = detector.detect_faces(image)

        # PRINT NUMBER OF FACES DETECTED
        logging.info(f'Faces Detected: {len(results)}')
        
        print('Face Detected: ', len(results))

        count = 0

        # PROCESS EACH DETECTED FACE
        for result in results:

            # GET BOUNDING BOX COORDINATES
            bounding_box = result['box']

            # PRINT BOUNDING BOX
            logging.info(f'Bounding Box: {bounding_box}')

            print(bounding_box)

            # GET CONFIDENCE SCORE
            confidence = result['confidence']

            # PRINT CONFIDENCE SCORE
            logging.info(f'Confidence Score: {confidence}')

            print(confidence)

            # CROP FACE IF CONDITIONS ARE MET
            if len(results) < 2 or confidence > 0.95:

                # SET MARGINS FOR CROPPING
                margin_x = bounding_box[2] * 0.3  # 30% AS THE MARGIN

                margin_y = bounding_box[3] * 0.3  # 30% AS THE MARGIN

                # CALCULATE CROPPING COORDINATES
                x1 = int(bounding_box[0] - margin_x)

                if x1 < 0:

                    x1 = 0

                x2 = int(bounding_box[0] + bounding_box[2] + margin_x)

                if x2 > image.shape[1]:

                    x2 = image.shape[1]

                y1 = int(bounding_box[1] - margin_y)

                if y1 < 0:

                    y1 = 0

                y2 = int(bounding_box[1] + bounding_box[3] + margin_y)

                if y2 > image.shape[0]:

                    y2 = image.shape[0]

                # PRINT CROPPING COORDINATES
                logging.info(f'Cropping Coordinates: {x1}, {y1}, {x2}, {y2}')

                print(x1, y1, x2, y2)

                # CROP THE IMAGE
                crop_image = image[y1:y2, x1:x2]

                # CREATE NEW FILENAME FOR THE CROPPED IMAGE
                new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, get_filename_only(frame)), count)

                count += 1

                # SAVE THE CROPPED IMAGE
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))

            else:

                # PRINT MESSAGE FOR SKIPPED FACES
                logging.info('Skipped a face due to low confidence or multiple faces detected.')

                print('Skipped a face..')
