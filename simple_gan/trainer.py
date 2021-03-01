from simple_gan.data_loader import DataLoader
from simple_gan.models import build_discriminator
from simple_gan.models import build_generator

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import numpy as np

class Trainer:

    def __init__(self, image_dir, image_width):
        data_loader = DataLoader(image_dir = image_dir,
                image_width = image_width)
        self.images = data_loader.load()
        self.input_shape = (image_width, image_width, 3)


    def _prepare_model(self, epochs, lr):

        discriminator = build_discriminator(input_shape = self.input_shape)
        generator = build_generator(input_shape = self.input_shape)
        disc_opt = Adam(lr = lr, beta_1 = 0.5, decay = lr / epochs)
        discriminator.compile(loss = "binary_crossentrophy", optimizer = disc_opt)
        discriminator.trainable = False
        gan_input = Input(shape = (100,))
        gan_output = generator(gan_input)

        gan = Model(gan_input, gan_output)
        gan_opt = Adam(lr = lr, beta_1 = 0.5, decay = lr / epochs)
        gan.compile(loss = "binary_crossentropy", optimizer = gan_opt)

        return gan

    def _get_batch_size(self):
        return int((25 * len(self.images))/ 100)





    def train(self, epochs = 1000, lr = 2e-4):

        self.images = (self.images.astype("float") - 127.5) / 127.5
        gan = self._prepare_model(epochs = epochs, lr = lr)
        benchmark_noise = np.random.uniform(-1, 1, size = (2156, 100))

        batch_size = self._get_batch_size()

        for epoch in range(0, epochs):
            batches_per_epoch = int(self.images.shape[0] / batch_size)

            for i in range(0, batches_per_epoch):
                p = None
                image_batch = self.images[i * batch_size : (i + 1) * batch_size]
                noise = np.random.uniform(-1, 1, size = (batch_size, 100))
                gan_image = gan.predict(noise, verbose = 0)
                X = np.concatenate((image_batch, gan_image))
                print(X)







