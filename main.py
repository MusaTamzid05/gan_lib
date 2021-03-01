from simple_gan.trainer import Trainer
from simple_gan.data_loader import DataLoader


def main():

    trainer = Trainer(image_dir = "./data", image_width = 224)
    trainer.train()



if __name__ == "__main__":
    main()
