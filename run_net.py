from shanghai_dataset import ShanghaiDataset

import torch.utils.data

def main():
    trainset = ShanghaiDataset("data/ShanghaiTech/A/train/")
    testset = ShanghaiDataset("data/ShanghaiTech/A/test/")

    print(testset[0]["count"])


if __name__ == '__main__':
    main()
