from simple_gan.trainer import Trainer


def main():

    trainer = Trainer(image_dir = "./data", image_width = 224)
    trainer.train()



if __name__ == "__main__":
    main()
