""" U_NET:
The U-Net model is a popular architecture for image segmentation tasks,
where the goal is to classify each pixel in an image as belonging to one of
several classes. The model consists of an encoder that downsamples the input
image to extract high-level features, a bridge that applies dilated
convolutionsto increase the receptive field, and a decoder that upsamples
the features to produce a segmentation mask. Skip connections are used to
combine low-level features from the encoder with high-level features from
the decoder, which helps to improve the accuracy of the segmentation.
"""


# The necessary modules from TensorFlow Keras are imported,
# including layers like Conv2D, Activation, and Input
# and also tf.model to build a model


from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D

from tensorflow.keras.models import Model

# importing a pre_trained model named ResNet50 model.
from tensorflow.keras.applications import ResNet50


def residual_block(inputs, num_filters):
    """residual_block:
    residual_block is function, which creates
    a residual_block as used in ResNet. It consists of two
    3x3 convolutional layers with batch normalization and
    ReLU activation, followed by a shortcut connection that
    adds the input to the output of the second convolutional layer.
    """
    # Conv2D layer with 3x3 kernel, same padding, and ReLU activation
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # Conv2D layer with 3x3 kernel, same padding, and no activation
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    # Shortcut connection with 1x1 kernel, same padding, and ReLU activation
    s = Conv2D(num_filters, 1, padding="same")(inputs)
    s = BatchNormalization()(s)
    x = Activation("relu")(x+s)

    return x


def dilated_conv(inputs, num_filters):
    """
    A function called dilated_conv is defined,
    which applies three dilated convolutions with
    dilation rates of 3, 6, and 9 to the input,
    and then concatenates the outputs and applies
    a 1x1 convolution with batch normalization
    and ReLU activation.
    """
    # Conv2D layer with 3x3 kernel, same padding,
    # and ReLU activation, dilation rate 3
    x1 = Conv2D(num_filters, 3, padding="same", dilation_rate=3)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    # Conv2D layer with 3x3 kernel, same padding,
    # and ReLU activation, dilation rate 6
    x2 = Conv2D(num_filters, 3, padding="same", dilation_rate=6)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    # Conv2D layer with 3x3 kernel, same padding,
    # and ReLU activation, dilation rate 9
    x3 = Conv2D(num_filters, 3, padding="same", dilation_rate=9)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)

    # Concatenate the three dilated convolution outputs
    x = Concatenate()([x1, x2, x3])

    # Conv2D layer with 1x1 kernel, same padding, and ReLU activation
    x = Conv2D(num_filters, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

# Define a decoder block function


def decoder_block(inputs, skip_features, num_filters):
    """
    A function called decoder_block is defined,
    which upsamples the input using bilinear interpolation,
    concatenates it with skip features from the encoder,
    and then applies a residual block.
    """
    # UpSampling2D layer with bilinear interpolation
    x = UpSampling2D((2, 2), interpolation="bilinear")(inputs)

    # Concatenate the upsampled input with the skip features
    x = Concatenate()([x, skip_features])

    # Residual block
    x = residual_block(x, num_filters)
    return x

# Define a function to build the U-Net model


def build_model(input_shape):
    """
    A function called build_model is defined, which builds the U-Net model.
    It takes an input shape as an argument and returns a Keras Model object.
    The input layer is defined with the given input shape.

    A pre-trained ResNet50 model is loaded and its output is used as the
    input to the U-Net model. The encoder consists of the first five
    convolutional blocks of ResNet50, with skip connections to the decoder.
    The bridge consists of a dilated convolution with 1024 filters.
    The decoder consists of four decoder blocks, each of which upsamples the
    input and concatenates it with skip features from the encoder,
    followed by a residual block. Four output layers are defined,
    each of which applies a 1x1 convolution with sigmoid activation
    to produce a binary segmentation mask. The output masks are concatenated
    along the channel dimension.

    The script is executed by calling the build_model function with
    an input shape of (512, 512, 3), which corresponds to a 512x512 RGB image.
    The resulting model is then summarized using the model.summary() method.
    """
    # Input
    inputs = Input(input_shape)

    # Pre-trained ResNet50 Model
    resnet50 = ResNet50(include_top=False,
                        weights="imagenet", input_tensor=inputs)

    # Encoder

    s1 = resnet50.get_layer("input_1").output
    # s1 = resnet50.get_layer("input_layer").output #for special tensorflow lib
    s2 = resnet50.get_layer("conv1_relu").output
    s3 = resnet50.get_layer("conv2_block3_out").output
    s4 = resnet50.get_layer("conv3_block4_out").output
    s5 = resnet50.get_layer("conv4_block6_out").output
    # print(s1.shape, s2.shape, s3.shape, s4.shape, s5.shape)

    # Bridge
    b1 = dilated_conv(s5, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    # print(d1.shape, d2.shape, d3.shape, d4.shape)

    y1 = UpSampling2D((8, 8), interpolation="bilinear")(d1)
    y1 = Conv2D(1, 1, padding="same", activation="sigmoid")(y1)

    y2 = UpSampling2D((4, 4), interpolation="bilinear")(d2)
    y2 = Conv2D(1, 1, padding="same", activation="sigmoid")(y2)

    y3 = UpSampling2D((2, 2), interpolation="bilinear")(d3)
    y3 = Conv2D(1, 1, padding="same", activation="sigmoid")(y3)

    y4 = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    outputs = Concatenate()([y1, y2, y3, y4])

    model = Model(inputs, outputs, name="U-2-Net")
    return model
