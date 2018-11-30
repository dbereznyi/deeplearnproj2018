from shanghai_dataset import ShanghaiDataset
from twice import TwICE

import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import math
import time

def main():
    transform = transforms.ToTensor()

    trainset = ShanghaiDataset("data/ShanghaiTech/A/train/", transform=transform)
    net = train_net(trainset, max_epochs=1)

    net_filename = "network_{}.pt".format(time.time())
    torch.save(net.state_dict(), net_filename)
    print("Saved network to {}.".format(net_filename))

    # net = TwICE()
    # net.load_state_dict(torch.load("network_1543557277.915378.pt"))

    testset = ShanghaiDataset("data/ShanghaiTech/A/test/", transform=transform)
    test_net(testset, net)


def train_net(trainset, max_epochs=10):
    """
    Trains a new instance of TwICE from scratch.
    :return: the trained network
    """
    # Use CUDA GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

    net = TwICE()
    net.apply(init_weights)
    net = net.to(device)

    criterion = torch.nn.MSELoss()

    learn_rate = 0.000001
    optimizer = torch.optim.Rprop(net.parameters(), lr=learn_rate)

    print("Beginning training with LR={} and {} epoch(s).\n".format(learn_rate, max_epochs))

    num_samples = len(trainset)
    for epoch in range(max_epochs):
        for i, (count, image) in enumerate(trainloader, 1):
            count = count.to(device).float()
            image = image.to(device)

            optimizer.zero_grad()

            time_start = time.perf_counter()
            net_count = net(image)
            time_end = time.perf_counter()

            print("[E#{} {}/{}] Output: {}, Actual: {}\n({} sec)\n"
                  .format(epoch + 1, i, num_samples, net_count.item(), count.item(), time_end - time_start))

            loss = criterion(net_count, count)
            loss.backward()

            optimizer.step()

    return net


def init_weights(model):
    if isinstance(model, torch.nn.Conv2d):
        torch.nn.init.normal_(model.weight.data, std=0.1)
        torch.nn.init.normal_(model.bias.data, std=0.1)


def test_net(testset, net):
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    mse = 0.0  # Mean squared error

    print("Beginning network evaluation on test dataset.\n")

    num_samples = len(testset)
    with torch.no_grad():
        for i, (count, image) in enumerate(testloader, 1):
            net_count = net(image)

            print("[{}/{}] Output: {}, Actual: {}\n".format(i, num_samples, net_count.item(), count.item()))

            mse += math.pow(count.float() - net_count, 2)

    mse /= num_samples

    print("MSE: {}".format(mse))

if __name__ == '__main__':
    main()
