from shanghai_dataset import ShanghaiDataset
from twice import TwICE

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

def main():
    trainset = ShanghaiDataset("data/ShanghaiTech/A/train/", transform=transforms.ToTensor())
    net = train_net(trainset, max_epochs=1)

    testset = ShanghaiDataset("data/ShanghaiTech/A/test/", transform=transforms.ToTensor())
    test_net(testset, net)


def train_net(trainset, max_epochs=10):
    """
    Trains a new instance of TwICE from scratch.
    :return: the trained network
    """
    # Use CUDA GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

    net = TwICE()
    net = net.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Rprop(net.parameters())

    num_samples = len(trainset)
    for epoch in range(max_epochs):
        for i, (count, image) in enumerate(trainloader):
            count = count.to(device)
            image = image.to(device)

            optimizer.zero_grad()

            net_count = net(image)

            print("[{}/{}] Output: {}, Actual: {}\n".format(i, num_samples, net_count.item(), count.item()))

            loss = criterion(net_count, torch.tensor(float(count)))
            loss.backward()

            optimizer.step()

    return net


def test_net(testset, net):
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

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
