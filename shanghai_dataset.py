from typing import Dict

import torch.utils.data
import csv
import os
import os.path
import skimage.io
import skimage.color


class ShanghaiDataset(torch.utils.data.Dataset):
    counts: Dict[int, int]

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # TODO load in data from both A and B and treat them as one dataset

        self.counts = {}
        with open(os.path.join(root_dir, "counts.csv"), newline="") as f:
            for (img_id, count) in csv.reader(f):
                self.counts[int(img_id)] = int(float(count))

        self.image_filenames = {}
        image_dir = os.path.join(root_dir, "images")
        for filename in os.listdir(image_dir):
            img_id = int(os.path.splitext(filename)[0].split("_")[-1])
            self.image_filenames[img_id] = filename

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        filepath = os.path.join(self.root_dir, "images", self.image_filenames[idx + 1])
        image = skimage.io.imread(filepath)
        image = skimage.color.gray2rgb(image)
        count = self.counts[idx + 1]

        if self.transform:
            image = self.transform(image)

        return count, image
