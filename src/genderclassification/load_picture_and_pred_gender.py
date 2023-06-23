import cv2
import numpy as np
import torch

from src.genderclassification.train_and_save import NeuralNetwork
from src.utils.showimg.show_images import show_image

if __name__ == "__main__":
    image = cv2.imread("myphotos/1.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    show_image(image)
    res = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    res = np.reshape(res, (3, 64, 64))
    model = NeuralNetwork()
    model.load_state_dict(torch.load("savedmodels/last"))
    model.eval()
    pred = model(torch.unsqueeze(torch.from_numpy(res), 0).to('cpu'))
    print(pred)
    if pred.argmax().item() == 1:
        print('you are a male')
        # print('jasteś chłopem')
    else:
        print('you are a female')
        # print('jesteś babą')
