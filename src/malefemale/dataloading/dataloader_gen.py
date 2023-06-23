import os

import torch
from torch.utils.data import Dataset, DataLoader
import json # we need to import json file with key points coordinates
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


class PerDirectoryImgClassificationDataset(Dataset):
    def __init__(self, directories=["dataset/female", "dataset/male"]):
        self.directories = directories
        self.img_paths_labels = []
        for idx, directory in enumerate(directories):
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    self.img_paths_labels.append((f, idx))

        random.shuffle(self.img_paths_labels)
    def __len__(self):
        return len(self.img_paths_labels)

    def __getitem__(self, i):
        image = cv2.imread(self.img_paths_labels[i][0])
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        res = cv2.resize(image, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
        res = np.reshape(res, (3, 300, 300))
        label = self.img_paths_labels[i][1]
        return res, label
