from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose



def discriminator_size_fixer(model, input_shape, min_size = 28):
    # for lack of a better name.
    if input_shape[0] != input_shape[1]:
        raise RuntimeError("image width and height must be same")
    neuron = 32
    if input_shape[0] == min_size:
        return model, neuron
    current_width = min_size
    neuron = 32
    while current_width > min_size:
        model.add(Conv2D(neuron, (5,5), padding = "same"), strides = 2)
        neuron *= 2
        current_width -= min_size
    return model, neuron


def build_discriminator(input_shape, alpha = 0.2):


    model = Sequential()
    model, neuron  = discriminator_size_fixer(model = model, input_shape = input_shape)
    model.add(Conv2D(neuron, (5,5), padding = "same",
                     strides = (2,2), input_shape = input_shape))
    model.add(LeakyReLU(alpha = alpha))
    model.add(Conv2D(neuron * 2, (5,5), padding = "same",
                     strides = (2,2), input_shape = input_shape))
    model.add(LeakyReLU(alpha = alpha))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha = alpha))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model


def generator_size_fixer(model, input_shape):
    current_width = 28
    target_width = input_shape[0]
    while current_width < target_width:
        model.add(Conv2DTranspose(input_shape[2], (5, 5), strides= (2, 2), padding = "same"))
        current_width *= 2

        if current_width != target_width:
            model.add(Activation("relu"))
    model.add(Activation("tanh"))
    return model

def build_generator(input_shape):
    if input_shape[0] != input_shape[1]:
        raise RuntimeError("input shape width and height must be same")

    middle_layer_shape = (7, 7, 64)
    model = Sequential()
    model.add(Dense(input_dim = 500, units = 512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dense(middle_layer_shape[0] * middle_layer_shape[1] * middle_layer_shape[2]))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Reshape(middle_layer_shape))
    model.add(Conv2DTranspose(32, (5, 5), strides= (2, 2), padding = "same"))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(input_shape[2], (5, 5), strides = (2, 2), padding = "same"))
    model = generator_size_fixer(model = model, input_shape = input_shape)
    return model

