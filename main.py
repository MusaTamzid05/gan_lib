from simple_gan.models import build_discrimator

def main():
    discriminator = build_discrimator(input_shape = (28, 28, 3))
    print(discriminator.summary())


if __name__ == "__main__":
    main()
