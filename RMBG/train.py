"""
this is a Python script that uses TensorFlow and Keras to build and
train a deep learning model for image segmentation. importing necessary
libraries, including TensorFlow, Keras, OpenCV, NumPy, Pandas, and some
utility functions like glob and os.
"""
import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from RMBG.model import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


class train:
    def __init__(self, IMG_H, IMG_W, BATCH, LR, EPOCH, SPLIT, PATH) -> None:
        # Set the TensorFlow log level to suppress unnecessary output
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        self.IMG_H = IMG_H  # 16
        self.IMG_W = IMG_W  # 16

        self.INPUT_SHAPE = (self.IMG_H, self.IMG_W, 3)
        self.BATCH = BATCH  # 32
        self.LR = LR  # 0.1
        self.EPOCH = EPOCH  # 5
        self.SPLIT = SPLIT

        # Set paths
        self.PATH = PATH  # r"D:\git ex\DA\RMBG-U2NET\Data"
        file = self.PATH + "\\file"

        self.MODEL_PATH = os.path.join(file, "model.h5")
        self.CSV_PATH = os.path.join(file, "data.csv")

    def GPU_setup():
        tf.debugging.set_log_device_placement(True)
        print(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    """
    Defining helper functions: The script following scripts and functions
    defines several helper functions to perform tasks like split dataset
    into training and validation sets, creating a directory, loading a dataset,
    reading an image, reading a mask, parsing a dataset using TensorFlow,
    and creating a TensorFlow dataset.
    """

    def load_dataset(self):
        # Loading the images and masks
        X = sorted(glob(os.path.join(self.PATH, "images", "*.jpg")))
        Y = sorted(glob(os.path.join(self.PATH, "masks", "*.png")))

        """
        Define a helper function to load the dataset
        and split it into training and validation sets
        """
        # split_size = int(len(X) * self.SPLIT)

        train_x, valid_x, train_y, valid_y = train_test_split(
            X, Y, train_size=1-self.SPLIT, test_size=self.SPLIT, random_state=42)

        return (train_x, train_y), (valid_x, valid_y)

    # Define a helper function to read an image and resize it

    def read_image(self, path):
        # Read the image and resize it
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (self.IMG_W, self.IMG_H))
        x = x/255.0
        x = x.astype(np.float32)
        return x

    # Define a helper function to read a mask and resize it

    def read_mask(self, path):
        # Read the mask and resize it
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (self.IMG_W, self.IMG_H))
        x = x.astype(np.float32)  # (h, w)
        x = np.expand_dims(x, axis=-1)  # (h, w, 1)
        x = np.concatenate([x, x, x, x], axis=-1)  # (h, w, 4)
        return x

    # Define a helper function to parse a dataset using TensorFlow

    def tf_parse(self, x, y):
        def _parse(x, y):
            x = self.read_image(x)
            y = self.read_mask(y)
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
        x.set_shape([self.IMG_W, self.IMG_H, 3])
        y.set_shape([self.IMG_W, self.IMG_H, 4])
        return x, y

    # Define a helper function to create a TensorFlow dataset

    def tf_dataset(self, X, Y, batch=2):
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        ds = ds.map(self.tf_parse).batch(batch).prefetch(10)
        return ds

    def train_model(self):
        # Set the random seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        file = self.PATH + "\\file"
        print(file)
        if not os.path.exists(file):
            os.makedirs(file)

        # Load the dataset
        (train_x, train_y), (valid_x, valid_y) = self.load_dataset()
        print(
            f"Train: {len(train_x)}/{len(train_y)} - Valid:{len(valid_x)}/{len(valid_y)}\n")

        """
        now we need to create TensorFlow datasets for
        both the training and validation sets, which are
        used to feed data into the model during training.
        """
        train_ds = self.tf_dataset(train_x, train_y, batch=self.BATCH)
        valid_ds = self.tf_dataset(valid_x, valid_y, batch=self.BATCH)

        """
        now we need to builds the deep learning model using
        the build_model function, which is defined elsewhere.
        The model is compiled with a binary cross-entropy
        loss function and an Adam optimizer with
        a specified learning rate.
        """
        model = build_model(self.INPUT_SHAPE)
        model.summary()
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(self.LR)
        )

        """
        The script defines several callbacks to be used during training,
        including model checkpointing, learning rate reduction,
        CSV logging, and early stopping.
        """
        callbacks = [
            ModelCheckpoint(
                self.MODEL_PATH, monitor='val_loss',
                verbose=1, save_best_only=True),

            ReduceLROnPlateau(
                monitor='val_loss', factor=0.1,
                patience=5, min_lr=1e-7, verbose=1),

            CSVLogger(self.CSV_PATH, append=True),
            EarlyStopping(
                monitor='val_loss', patience=20,
                restore_best_weights=False)
        ]

        """
        Finally, the script trains the model using the fit method,
        passing in the training and validation datasets and
        the callbacks defined earlier.
        """
        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=self.EPOCH,
            callbacks=callbacks
        )
