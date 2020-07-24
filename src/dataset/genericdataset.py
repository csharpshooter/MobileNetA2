import os
import cv2
import numpy as np
import torch.utils.data
from PIL import Image
from tqdm import trange, tqdm
import asyncio
import nest_asyncio
from PIL import ImageFile
import torchdata as td


class GenericDataset(td.Dataset):
    def __init__(self, images, labels, transforms, classes=None, preload=False):
        super().__init__()
        # load all image files, sorting them to
        # ensure that they are aligned
        self.images = images
        self.labels = labels
        self.preload = preload
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.image_memory = []
        self.transforms = transforms
        self.classes = classes

        # for idx in tqdm(self.images):
        #     img = cv2.imread(idx)
        #     self.image_memory.append(img)

        #
        # if self.preload:
        #     self.preload_image()
        #     asyncio.ensure_future(self.preload_image())

    def __getitem__(self, idx):
        if self.preload:
            image = self.image_memory[idx].convert("RGB")
        else:
            image = Image.open(self.images[idx]).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        label = {
            'Flying Birds': 0,
            'Large QuadCopters': 1,
            'Small QuadCopters': 2,
            'Winged Drones': 3
        }

        return image, label.get(self.labels[idx])

    def __len__(self):
        return len(self.images)

    async def preload_image(self):
        print("\nPreloading images from dataset...")
        # async for idx in AsyncIterator(tqdm(self.images)):
        for idx in tqdm(self.images):
            img = cv2.imread(idx)
            self.image_memory.append(img)

    def set_transforms(self, transforms=None):
        self.transforms = transforms


class AsyncIterator:
    def __init__(self, seq):
        self.iter = iter(seq)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration
