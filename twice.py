import torch.nn as nn
import torch.nn.functional as func


class TwICE(nn.Module):
    """
    The "Three-way Integrated Crowd Estimator", or "TwICE".
    """
    def __init__(self):
        super(TwICE, self).__init__()

        self.branches = nn.ModuleList([
            ConvInt(12, 9),
            ConvInt(9, 7),
            ConvInt(7, 5)
        ])

    def forward(self, x):
        print("img_dims: {}".format(x.size()))

        counts = [branch.forward(x.clone()) for branch in self.branches]

        print("counts = {}".format([count.item() for count in counts]))

        return sum(counts) / len(counts)


class ConvInt(nn.Module):
    """
    A fully-convolutional network similar, taken from "Fully Convolutional Crowd Counting On Highly Congested Scenes",
    but with parameters for controlling the size of the first and middle kernels.

    Used as the branches of TwICE.
    """
    def __init__(self, init_kernel, mid_kernel):
        super(ConvInt, self).__init__()

        self.conv_inits = nn.ModuleList([
            nn.Conv2d(3, 36, init_kernel),
            nn.Conv2d(36, 72, mid_kernel)
        ])

        self.conv_mids = nn.ModuleList([
            nn.Conv2d(72, 36, mid_kernel),
            nn.Conv2d(36, 24, mid_kernel),
            nn.Conv2d(24, 16, mid_kernel),
            nn.Conv2d(16, 1, 1)
        ])

    def forward(self, x):
        for layer in self.conv_inits:
            x = func.relu(layer(x))
            x = func.max_pool2d(x, 2, stride=2)

        for layer in self.conv_mids:
            x = func.relu(layer(x))

        y = x.view(x.size()[2] * x.size()[3])

        return sum(y)