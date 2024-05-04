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
from model import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Set the TensorFlow log level to suppress unnecessary output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Define global variables for image height and width
global image_h
global image_w

# Define a helper function to create a directory


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


"""
Defining helper functions: The script following scripts and functions
defines several helper functions to perform tasks like split dataset
into training and validation sets, creating a directory, loading a dataset,
reading an image, reading a mask, parsing a dataset using TensorFlow,
and creating a TensorFlow dataset.
"""


def load_dataset(path, split=0.4):
    # Loading the images and masks
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    """
    Define a helper function to load the dataset
    and split it into training and validation sets
    """
    split_size = int(len(X) * split)

    train_x, valid_x = train_test_split(
        X, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(
        Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)

# Define a helper function to read an image and resize it


def read_image(path):
    # Read the image and resize it
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    x = x/255.0
    x = x.astype(np.float32)
    return x

# Define a helper function to read a mask and resize it


def read_mask(path):
    # Read the mask and resize it
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (image_w, image_h))
    x = x.astype(np.float32)  # (h, w)
    x = np.expand_dims(x, axis=-1)  # (h, w, 1)
    x = np.concatenate([x, x, x, x], axis=-1)  # (h, w, 4)
    return x

# Define a helper function to parse a dataset using TensorFlow


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([image_h, image_w, 3])
    y.set_shape([image_h, image_w, 4])
    return x, y

# Define a helper function to create a TensorFlow dataset


def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds


if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Set the random seed for reproducibility
    create_dir("files")

    # Set hyperparameters
    image_h = 512
    image_w = 512
    input_shape = (image_h, image_w, 3)
    batch_size = 4
    lr = 1e-4
    num_epochs = 100

    # Set paths
    dataset_path = " "
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    # Load the dataset
    (train_x, train_y), (valid_x, valid_y) =\
        load_dataset(dataset_path, split=0.2)
    print(
        f"Train: {len(train_x)}/{len(train_y)} - Valid:\
            {len(valid_x)}/{len(valid_y)}\n")

    """
    now we need to create TensorFlow datasets for
    both the training and validation sets, which are
    used to feed data into the model during training.
    """
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

    """
    now we need to builds the deep learning model using
    the build_model function, which is defined elsewhere.
    The model is compiled with a binary cross-entropy
    loss function and an Adam optimizer with
    a specified learning rate.
    """
    model = build_model(input_shape)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )

    """
    The script defines several callbacks to be used during training,
    including model checkpointing, learning rate reduction,
    CSV logging, and early stopping.
    """
    callbacks = [
        ModelCheckpoint(
            model_path, monitor='val_loss',
            verbose=1, save_best_only=True),

        ReduceLROnPlateau(
            monitor='val_loss', factor=0.1,
            patience=5, min_lr=1e-7, verbose=1),

        CSVLogger(csv_path, append=True),
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
        epochs=num_epochs,
        callbacks=callbacks
    )
