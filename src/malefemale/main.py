import torch
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

if __name__ == "__main__":
    print("gpu: ", str(torch.cuda.is_available()))
    print("count: ", str(torch.cuda.device_count()))
    device_id = torch.cuda.current_device()
    print("device id: ", str(device_id))
    print("device name: ", str(torch.cuda.get_device_name(device_id)))

    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder('dataset/female/', transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    images, labels = next(iter(dataloader))
    plt.imshow(images[0], normalize=False)
