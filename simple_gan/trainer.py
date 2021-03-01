from simple_gan.data_loader import DataLoader
from simple_gan.models import build_discriminator
from simple_gan.models import build_generator

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.utils import shuffle

import numpy as np
import os
import cv2


class Trainer:

    def __init__(self, image_dir, image_width, save_dir = "results"):
        data_loader = DataLoader(image_dir = image_dir,
                image_width = image_width)
        self.images = data_loader.load()
        self.input_shape = (image_width, image_width, 3)

        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)

        self.save_dir = save_dir


    def _prepare_model(self, epochs, lr):

        discriminator = build_discriminator(input_shape = self.input_shape)
        generator = build_generator(input_shape = self.input_shape)
        disc_opt = Adam(lr = lr, beta_1 = 0.5, decay = lr / epochs)
        discriminator.compile(loss = "binary_crossentropy", optimizer = disc_opt)
        discriminator.trainable = False
        gan_input = Input(shape = (100,))
        gan_output = discriminator(generator(gan_input))

        gan = Model(gan_input, gan_output)
        gan_opt = Adam(lr = lr, beta_1 = 0.5, decay = lr / epochs)
        gan.compile(loss = "binary_crossentropy", optimizer = gan_opt)

        return gan, discriminator, generator

    def _get_batch_size(self):
        return int((25 * len(self.images))/ 100)

    def save_results(self, epoch, generator, benchmark_noise):
        dst_dir = os.path.join(self.save_dir, str(epoch))
        os.mkdir(dst_dir)

        images = generator.predict(benchmark_noise)
        images = ((images * 127.5) + 127.5).astype("uint8")

        for i, image in enumerate(images):
            dst_path = os.path.join(dst_dir, f"{i}.jpg")
            cv2.imwrite(dst_path, image)





    def train(self, epochs = 1000, lr = 2e-4):

        self.images = (self.images.astype("float") - 127.5) / 127.5
        gan, discriminator, generator = self._prepare_model(epochs = epochs, lr = lr)
        benchmark_noise = np.random.uniform(-1, 1, size = (10, 100))

        batch_size = self._get_batch_size()

        for epoch in range(0, epochs):
            batches_per_epoch = int(self.images.shape[0] / batch_size)

            disc_losses = []
            gen_losses = []

            for i in range(0, batches_per_epoch):
                image_batch = self.images[i * batch_size : (i + 1) * batch_size]
                noise = np.random.uniform(-1, 1, size = (batch_size, 100))

                # generate random image by generator
                gen_images = generator.predict(noise, verbose = 0)
                X = np.concatenate((image_batch, gen_images))
                y =  ([1] * batch_size) + ([0] * batch_size)
                y = np.reshape(y, (-1,))
                (X, y) = shuffle(X, y)


                # train the discriminator
                disc_loss = discriminator.train_on_batch(X, y)
                disc_losses.append(disc_loss)

                # train the gan

                noise = np.random.uniform(-1, 1, (batch_size, 100))
                fake_labels = [1] * batch_size
                fake_labels = np.reshape(fake_labels, (-1,))
                gen_loss = gan.train_on_batch(noise, fake_labels)
                gen_losses.append(gen_loss)


                if i == batches_per_epoch - 1:
                    self.save_results(epoch = epoch, generator =  generator, benchmark_noise =  benchmark_noise)


            print(f"epoch : {epoch + 1} disc loss {disc_loss / batches_per_epoch}, generator loss : {gen_loss / batches_per_epoch}")


