"""
this file is to use a trained model
to make predictions on a set of images.
"""


# Import necessary libraries and modules
import os
import cv2
import numpy as np
import tensorflow as tf


from tqdm import tqdm
from glob import glob
from train import create_dir

# Set environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Global parameters
""" Global parameters """
image_h = 512   # Height of the images
image_w = 512   # Width of the images


if __name__ == "__main__":
    # Seeding for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create directory for storing output masks
    create_dir("test/masks")

    # Load the trained model
    model = tf.keras.models.load_model("files/model.h5")

    # Load the dataset
    data_x = glob("test/images/*")

    # Loop over each image file
    for path in tqdm(data_x, total=len(data_x)):
        # Extract the file name
        name = path.split("/")[-1].split(".")[0]

        # Read the image using OpenCV
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        # Get the height and width of the original image
        h, w, _ = image.shape

        # Resize the image to the specified dimensions
        x = cv2.resize(image, (image_w, image_h))

        # Normalize the image by dividing by 255.0
        x = x/255.0

        # Convert the image to a float32 data type
        x = x.astype(np.float32)  # (h, w, 3)

        # Expand the image to a 4D tensor with a batch size of 1
        x = np.expand_dims(x, axis=0)  # (1, h, w, 3)

        # Make a prediction using the model
        y = model.predict(x, verbose=0)[0][:, :, -1]
        # Resize the prediction to the original image dimensions
        y = cv2.resize(y, (w, h))
        # Expand the prediction to a 3D tensor
        y = np.expand_dims(y, axis=-1)

        """
        Create a masked image by multiplying
        the original image with the prediction
        """
        masked_image = image * y

        # Create a line for visualization purposes
        line = np.ones((h, 10, 3)) * 128

        """
        Concatenate the original image,
        line, and masked image along the width axis
        """
        cat_images = np.concatenate([image, line, masked_image], axis=1)

        # Save the concatenated image to a file
        cv2.imwrite(f"test/masks/{name}.png", cat_images)
