from simple_gan.data_loader import DataLoader


def main():

    data_loader = DataLoader(image_dir = "./data", image_width = 224)
    data_loader.load()



if __name__ == "__main__":
    main()
