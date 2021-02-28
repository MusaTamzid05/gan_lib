import cv2
import os
import numpy as np

class DataLoader:

    def __init__(self, image_dir, image_width):

        self.image_paths = self._load_paths(image_dir = image_dir)
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image in {self.image_paths}")

        self.image_paths = sorted(self.image_paths)
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
        len_images = len(self.image_paths)

        for index, image_path in enumerate(self.image_paths):
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.image_shape)

            if images is None:
                images = np.array(image)
            else:
                images = np.append(images, image)
            self._show_progress(current_index = index, total = len_images )

        images = images.reshape((len(self.image_paths), self.image_shape[0], self.image_shape[1], 3))
        return images

    def _show_progress(self, current_index, total):

        current_index = current_index + 1
        total = total

        progress = int((current_index * 100 )/ total)
        print("loading images => [", end = "")

        for index in range(100):
            if index   < progress:
                print(">", end = "")
            else:
                print("-", end = "")

        print(f"]({current_index} / {total})")






