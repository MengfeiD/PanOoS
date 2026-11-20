import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class QuadOoS(Dataset):
    def __init__(self, hparams, transforms):
        super().__init__()
        self.hparam = hparams
        self.transforms = transforms

        images_root = os.path.join(hparams.dataset_root, 'images')
        labels_root = os.path.join(hparams.dataset_root, 'masks')
        self.images = os.listdir(images_root)
        self.labels = [os.path.join(labels_root, img_path.replace('jpg', 'png')) for img_path in self.images]
        self.images = [os.path.join(images_root, img_path) for img_path in self.images]

        self.num_samples = len(self.images)

    def __getitem__(self, index):
        
        image = np.array(Image.open(self.images[index]).convert('RGB'))
        label = np.array(Image.open(self.labels[index]))

        label[label == 254] = 1

        if self.transforms is not None:
            aug = self.transforms(image=image, mask=label)
            image = aug['image']
            label = aug['mask']

        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples