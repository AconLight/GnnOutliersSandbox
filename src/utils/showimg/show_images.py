import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def show_grid_images(images, title=""):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx])
                idx += 1
    fig.suptitle(title, fontsize=30)

    plt.show()

def show_image(image, title=""):
    image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if type(image) is torch.Tensor:
        image = image.detach().cpu().numpy()

    if image.shape[0] == 3:
        image = np.reshape(image, (image.shape[1], image.shape[2], 3))
    plt.imshow(image)
    plt.title(title)
    plt.show()