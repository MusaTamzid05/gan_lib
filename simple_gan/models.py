from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization



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

