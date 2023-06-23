import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

from src.utils.showimg.show_images import show_image


class PerDirectoryImgClassificationDataset(Dataset):
    def __init__(self, directories, normalize, size):
        self.normalize = normalize
        self.size = size
        self.directories = directories
        self.img_paths_labels = []
        for idx, directory in enumerate(directories):
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    self.img_paths_labels.append((f, idx))

        random.shuffle(self.img_paths_labels)
        self.imgs = [None] * len(self.img_paths_labels)
    def __len__(self):
        return len(self.img_paths_labels)

    def __getitem__(self, i):
        if self.imgs[i] is None:
            image = cv2.imread(self.img_paths_labels[i][0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.normalize:
                image = cv2.normalize(image, None, self.normalize[0], self.normalize[1], cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if self.size:
                res = cv2.resize(image, dsize=self.size, interpolation=cv2.INTER_CUBIC)
            res = np.reshape(res, (3, res.shape[0], res.shape[1]))
            self.imgs[i] = res
        label = self.img_paths_labels[i][1]
        return self.imgs[i], label
