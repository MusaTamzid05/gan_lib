import cv2
import os
import numpy as np

class DataLoader:

    def __init__(self, image_dir, image_width):

        self.image_paths = self._load_paths(image_dir = image_dir)
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image in {self.image_paths}")
        self.image_shape = (image_width, image_width)

    def _load_paths(self, image_dir):
        image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir) if self._is_valid_image_file(image_name)]
        return image_paths

    def _is_valid_image_file(self, filename):
        valid_ext = ["jpg", "png", "jpeg"]

        for ext in valid_ext:
            if filename.endswith(ext):
                return True

        return False

    def load(self):

        images = None

        for image_path in self.image_paths:
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.image_shape)

            if images is None:
                images = np.array(image)
            else:
                images = np.append(images, image)

        images = images.reshape((len(self.image_paths), self.image_shape[0], self.image_shape[1], 3))
        print(images.shape)




