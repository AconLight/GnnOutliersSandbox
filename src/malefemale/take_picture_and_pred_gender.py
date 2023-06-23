import cv2
import torch
from cv2 import VideoCapture
import numpy as np
from matplotlib import pyplot as plt

from src.malefemale.main import NeuralNetwork

if __name__ == "__main__":
    cam = VideoCapture(0)
    result, image = cam.read()
    if result:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        plt.imshow(image)
        plt.title('photo')
        plt.show()
        res = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        res = np.reshape(res, (3, 64, 64))
        model = NeuralNetwork()
        model.load_state_dict(torch.load("savedmodels/last"))
        model.eval()
        pred = model(torch.unsqueeze(torch.from_numpy(res), 0).to('cpu'))
        print(pred)
        if pred.argmax().item() == 1:
            print('you are male')
            # print('jasteś chłopem')
        else:
            print('you are female')
            # print('jesteś babą')

    else:
        print('sth went wrong')