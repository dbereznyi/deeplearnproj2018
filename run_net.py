from shanghai_dataset import ShanghaiDataset
from twice import TwICE

import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import math

def main():
    net = train_net()
    test_net(net)


def train_net(max_epochs=10):
    """
    Trains a new instance of TwICE from scratch.
    :return: the trained network
    """
    trainset = ShanghaiDataset("data/ShanghaiTech/A/train/", transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True, num_workers=2)

    net = TwICE()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(max_epochs):
        for count, image in trainloader:
            optimizer.zero_grad()

            net_count = net(image)
            loss = criterion(net_count, count)
            loss.backward()
            optimizer.step()

    # TODO finish training net

    return net


def test_net(net):
    testset = ShanghaiDataset("data/ShanghaiTech/A/test/", transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=3, shuffle=True, num_workers=2)

    # TODO finish

    avg_err = 0.0

    with torch.no_grad():
        for count, image in testloader:
            net_count = net(image)
            avg_err += abs(count - net_count) / count

    avg_err /= len(testset)

    print("Average error: {.5}%".format(avg_err * 100))

if __name__ == '__main__':
    main()
