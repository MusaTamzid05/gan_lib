#from simple_gan.models import build_discriminator
from simple_gan.models import build_generator

def main():
    #discriminator = build_discriminator(input_shape = (28, 28, 3))
    #print(discriminator.summary())
    generator = build_generator(input_shape = (28, 28, 3))
    generator.summary()


if __name__ == "__main__":
    main()
