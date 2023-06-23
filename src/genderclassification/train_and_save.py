import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.dataloading.dataloader_gen import PerDirectoryImgClassificationDataset
from src.utils.showimg.show_images import show_image

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

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3042, 1000),
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

        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


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
    female_path = Path(__file__).parents[1] / "datasets" / "malefemale" / "female"
    male_path = Path(__file__).parents[1] / "datasets" / "malefemale" / "male"
    dataset = PerDirectoryImgClassificationDataset([female_path, male_path], (0, 1), (64, 64))
    garbage, shrinked = torch.utils.data.random_split(dataset, [0.0, 1.0])
    train, test = torch.utils.data.random_split(shrinked, [0.7, 0.3])
    return train, test


if __name__ == "__main__":
    start = time.time()

    training_data, test_data = get_train_test()
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    print("images paths loaded: ", str(time.time() - start), "sec")
    start = time.time()

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print('device: ', device)

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("model loaded: ", str(time.time() - start), "sec")
    start = time.time()

    epochs = 500
    for i, t in enumerate(range(epochs)):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        print(("loading images + " if i == 0 else "") + "learning time: ", str(time.time() - start), "sec")
    print("Done!")

    example_img, example_label = test_dataloader.dataset[0]
    print('example label:', example_label)
    pred = model(torch.unsqueeze(torch.from_numpy(example_img), 0).to(device))
    print('pred label   :', pred.argmax().item())
    show_image(example_img)
    print('saving model')
    torch.save(model.state_dict(), "savedmodels/last")
