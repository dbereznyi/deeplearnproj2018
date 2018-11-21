from shanghai_dataset import ShanghaiDataset

import torch.utils.data

def main():
    trainset = ShanghaiDataset("data/ShanghaiTech/A/train/")
    testset = ShanghaiDataset("data/ShanghaiTech/A/test/")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=3, shuffle=True, num_workers=2)

    print(testset[0]["count"])


if __name__ == '__main__':
    main()
