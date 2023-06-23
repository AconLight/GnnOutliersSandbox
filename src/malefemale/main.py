import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import numpy as np

from src.malefemale.dataloading.dataloader_gen import PerDirectoryImgClassificationDataset

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.rel1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 18, 3)
        self.rel2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(18, 18, 3)
        self.rel3 = nn.ReLU()
        self.conv4 = nn.Conv2d(18, 18, 3)
        self.rel4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)


        # self.conv5 = nn.Conv2d(256, 512, 3)
        # self.rel5 = nn.ReLU()
        # self.conv6 = nn.Conv2d(512, 1024, 3)
        # self.rel6 = nn.ReLU()
        # self.maxpool3 = nn.MaxPool2d(2)


        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(93312, 1000),
            nn.ReLU(),
            nn.Linear(1000, 250),
            nn.ReLU(),
            nn.Linear(250, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.rel1(x)
        x = self.conv2(x)
        x = self.rel2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.rel3(x)
        x = self.conv4(x)
        x = self.rel4(x)
        x = self.maxpool2(x)

        # x = self.conv5(x)
        # x = self.rel5(x)
        # x = self.conv6(x)
        # x = self.rel6(x)
        # x = self.maxpool3(x)

        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # # if True:#batch != 0 and batch % 30 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:

            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def get_train_test():
    dataset = PerDirectoryImgClassificationDataset()
    garbage, shrinked = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train, test = torch.utils.data.random_split(shrinked, [0.7, 0.3])
    return train, test

if __name__ == "__main__":

    training_data, test_data = get_train_test()

    batch_size = 4

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Get cpu, gpu or mps device for training.
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print('device: ', device)

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

    epochs = 3
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    example_img, example_label = test_dataloader.dataset[0]
    print('example label:', example_label)
    plt.imshow(np.reshape(example_img, (300, 300, 3)))
    plt.title('example')
    plt.show()
    # pred = model(torch.from_numpy(example_img).to(device))
    # print('pred label   :', pred.argmax().item())

