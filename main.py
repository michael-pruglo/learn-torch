import torch, torchvision, torchvision.transforms as transforms
import numpy as np, matplotlib.pyplot as plt


def get_data(training):
    return torchvision.datasets.MNIST(
        root="./datasets/",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

def main():
    training_set_loader =   torch.utils.data.DataLoader(get_data(training=True), batch_size=10)
    validation_set_loader = torch.utils.data.DataLoader(get_data(training=False), batch_size=10)

    training_batch = next(iter(training_set_loader))
    images, labels = training_batch
    
    print(images.squeeze().shape)
    print("labels:", labels)
    grid = torchvision.utils.make_grid(images)
    plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()


if __name__ == "__main__":
    main()