import json

import os

from distutils.dir_util import copy_tree

import shutil

import numpy as np

import splitfolders as split_folders

from logger import logging

os.chdir('..')

# BASE PATH FOR TRAIN SAMPLE VIDEOS
base_path = 'train_sample_videos'

# PATH FOR PREPARED DATASET
dataset_path = 'prepared_dataset'

# CREATE PREPARED DATASET DIRECTORY
logging.info(f'Creating Directory: {dataset_path}')

print('Creating Directory: ' + dataset_path)

os.makedirs(dataset_path, exist_ok=True)

# TEMPORARY DIRECTORY FOR FAKE FACES
tmp_fake_path = 'tmp_fake_faces'

# CREATE TEMPORARY DIRECTORY FOR FAKE FACES
logging.info(f'Creating Directory: {tmp_fake_path}')

print('Creating Directory: ' + tmp_fake_path)

os.makedirs(tmp_fake_path, exist_ok=True)


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


# CREATE DIRECTORIES FOR REAL AND FAKE FACES
real_path = os.path.join(dataset_path, 'real')

# CREATE DIRECTORY FOR REAL FACES
logging.info(f'Creating Directory: {real_path}')

print('Creating Directory: ' + real_path)

os.makedirs(real_path, exist_ok=True)

fake_path = os.path.join(dataset_path, 'fake')

# CREATE DIRECTORY FOR FAKE FACES
logging.info(f'Creating Directory: {fake_path}')

print('Creating Directory: ' + fake_path)

os.makedirs(fake_path, exist_ok=True)


# PROCESS EACH FILENAME IN METADATA
for filename in metadata.keys():

    logging.info(f'Processing filename: {filename}')
    print(filename)

    # PRINT LABEL FOR CURRENT FILENAME
    label = metadata[filename]['label']

    logging.info(f'Label: {label}')

    print(label)

    tmp_path = os.path.join(os.path.join(base_path, get_filename_only(filename)), 'faces')

    logging.info(f'Temporary path: {tmp_path}')

    print(tmp_path)

    # CHECK IF FACES DIRECTORY EXISTS
    if os.path.exists(tmp_path):

        # COPY REAL FACES TO REAL DIRECTORY
        if label == 'REAL':    

            logging.info(f'Copying to: {real_path}')

            print('Copying to :' + real_path)

            copy_tree(tmp_path, real_path)

        # COPY FAKE FACES TO TEMP DIRECTORY
        elif label == 'FAKE':

            logging.info(f'Copying to: {tmp_fake_path}')

            print('Copying to :' + tmp_fake_path)

            copy_tree(tmp_path, tmp_fake_path)

        else:

            logging.info('Ignored..')
            print('Ignored..')


# LIST ALL REAL FACES
all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]

# PRINT TOTAL NUMBER OF REAL FACES
logging.info(f'Total Number of Real faces: {len(all_real_faces)}')

print('Total Number of Real faces: ', len(all_real_faces))


# LIST ALL FAKE FACES
all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]

# PRINT TOTAL NUMBER OF FAKE FACES
logging.info(f'Total Number of Fake faces: {len(all_fake_faces)}')

print('Total Number of Fake faces: ', len(all_fake_faces))


# RANDOMLY SELECT FAKE FACES EQUAL TO NUMBER OF REAL FACES
random_faces = np.random.choice(all_fake_faces, len(all_real_faces), replace=False)

# COPY SELECTED FAKE FACES TO FAKE DIRECTORY
for fname in random_faces:

    src = os.path.join(tmp_fake_path, fname)

    dst = os.path.join(fake_path, fname)

    shutil.copyfile(src, dst)

# PRINT MESSAGE FOR DOWN-SAMPLING COMPLETION
logging.info('Down-sampling Done!')

print('Down-sampling Done!')


# SPLIT INTO TRAIN, VALIDATION, AND TEST FOLDERS
split_folders.ratio(dataset_path, output='split_dataset', seed=1377, ratio=(1, 0, 0))  # DEFAULT VALUES

# PRINT MESSAGE FOR SPLITTING COMPLETION
logging.info('Train/ Val/ Test Split Done!')

print('Train/ Val/ Test Split Done!')
