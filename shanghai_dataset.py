from typing import Dict

import torch.utils.data
import csv
import os
import os.path
import skimage.io


class ShanghaiDataset(torch.utils.data.Dataset):
    counts: Dict[int, int]

    def __init__(self, root_dir):
        self.root_dir = root_dir

        # TODO load in data from both A and B and treat them into one dataset

        self.counts = {}
        with open(os.path.join(root_dir, "counts.csv"), newline="") as f:
            reader = csv.reader(f)
            for (img_id, count) in reader:
                self.counts[int(img_id)] = int(float(count))

        self.image_filenames = {}
        for fname in os.listdir(os.path.join(root_dir, "images")):
            img_id = int(os.path.splitext(fname)[0].split("_")[-1])
            self.image_filenames[img_id] = fname

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        # TODO maybe read in and cache images for better performance
        image = skimage.io.imread(os.path.join(self.root_dir, "images", self.image_filenames[idx + 1]))
        count = self.counts[idx + 1]

        return {
            "count": count,
            "image": image
        }
