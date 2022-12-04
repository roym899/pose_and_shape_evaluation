import torch.utils.data as data
import torchvision
from PIL import Image
import os
import numpy as np

class DistractorDataset(data.Dataset):
    def __init__(self, distractor_dir, chrom_rand_level, size_crop=(128, 128)):
        self.dis_fns = []
        if distractor_dir is not None:
            for fn in os.listdir(distractor_dir):
                self.dis_fns.append(os.path.join(distractor_dir, fn))
        self.num_dis = len(self.dis_fns)

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(360),
                torchvision.transforms.RandomResizedCrop(size_crop, scale=(0.08, 1.5)),
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            ]
        )

        self.chrom_rand_level = chrom_rand_level

    def __len__(self):
        return self.num_dis

    def __getitem__(self, idx):
        return self.load(self.dis_fns[idx])

    def load(self, fn):

        image_dis = Image.open(fn).convert("RGB")
        image_dis = self.transform(image_dis)
        distractor = np.array(image_dis)

        distractor = distractor.transpose(
            2, 0, 1).astype(
            np.float32) / 255.0



        return distractor
