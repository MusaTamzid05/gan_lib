from simple_gan.models import build_discriminator

def main():
    discriminator = build_discriminator(input_shape = (28, 28, 3))
    print(discriminator.summary())


if __name__ == "__main__":
    main()
