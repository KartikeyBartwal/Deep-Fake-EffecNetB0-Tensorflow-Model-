docs/Screenshot from 2024-08-08 03-05-46.png

=================
Deep-Fake-EffecNetB0-Tensorflow
=================


.. image:: https://img.shields.io/pypi/v/effnetb0_deep_learning.svg
        :target: https://pypi.python.org/pypi/effnetb0_deep_learning

.. image:: https://img.shields.io/travis/KartikeyBartwal/effnetb0_deep_learning.svg
        :target: https://travis-ci.com/KartikeyBartwal/effnetb0_deep_learning

.. image:: https://readthedocs.org/projects/effnetb0-deep-learning/badge/?version=latest
        :target: https://effnetb0-deep-learning.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


EfficientNetB0 Deep Learning is a Python package designed to facilitate the training and evaluation of deep learning models using the EfficientNetB0 architecture for image classification tasks.

* Free software: MIT license
* Documentation: https://effnetb0-deep-learning.readthedocs.io

Features
--------

* **Model Training**: Train models on custom datasets with the ability to fine-tune parameters.
  
* **Pre-trained Weights**: Utilize pre-trained weights for improved performance on image classification tasks.
  
* **Model Evaluation**: Assess model performance with various metrics including accuracy and loss graphs.
  
* **Logging**: Track training progress and performance metrics through detailed logs.
  
* **Custom Dataset Handling**: Easily manage datasets with support for structured training, validation, and testing splits.

Installation
============

To install the required packages, run:

.. code-block:: bash

    pip install -r requirements.txt

Usage
=====

Step 0 - Convert Video Frames to Individual Images
--------------------------------------------------

To extract all the video frames from the acquired deepfake datasets and save them as individual images for further processing, run:

.. code-block:: bash

    python 00-convert_video_to_image.py

Step 1 - Extract Faces from the Deepfake Images with MTCNN
-----------------------------------------------------------

To extract faces from the deepfake images, use the pre-trained MTCNN model from the following GitHub repository:

https://github.com/ipazc/mtcnn

Run the following command:

.. code-block:: bash

    python 01a-crop_faces_with_mtcnn.py

Step 2 - Balance and Split Datasets into Various Folders
--------------------------------------------------------

We need to split the dataset into training, validation, and testing sets (for example, in the ratio of 80:10:10) as the final step in the data preparation phase. Run:

.. code-block:: bash

    python 02-prepare_fake_real_dataset.py

Step 3 - Model Training
------------------------

We use EfficientNet as the backbone for model training, treating the deepfake detection task as a binary classification problem applicable to both video and image content.

In this code implementation, we modified the EfficientNet B0 model by:

-Changing the initial input layer to accept 128x128 images with a depth of 3.
-Connecting the last convolutional output to a global max pooling layer.
-Adding two fully connected layers with ReLU activations.
-Utilizing a Sigmoid activation in the output layer to act as a binary classifier.

Thus, the model is expected to produce an output between 0 and 1 for a colored square image, indicating the probability of the image being either a deepfake (0) or genuine (1).
Run the following command to start training:

.. code-block:: bash

    python 03-train_cnn.py
