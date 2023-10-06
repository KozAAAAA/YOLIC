"""Common code for the Yolic training and testing scripts"""

from typing import Any
from os import listdir

from PIL import Image

import torch
import torch.nn as nn

import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v2

# TODO: Move classes declaration here

NUMBER_OF_COIS = 104
NUMBER_OF_CLASSES = 11
YOLIC_NET_INPUT_WIDTH = 224
YOLIC_NET_INPUT_HEIGHT = 224

TRAIN_IMAGE_DIR = Path("./data/split_images/train/outdoor/")
VAL_IMAGE_DIR = Path("./data/split_images/val/outdoor/")
TEST_IMAGE_DIR = Path("./data/split_images/test/outdoor/")

LABEL_DIR = Path("./data/labels/")

YOLIC_MODEL_PATH = Path("yolic_model.pt")


# TODO: lacks data augmentation as in the original code
class YolicDataset(Dataset):
    """Yolic Dataset class used to load images and label"""

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        transform=None,
    ):
        """Initialize YolicDataset class"""
        self._image_dir = image_dir
        self._label_dir = label_dir
        self._image_names = listdir(image_dir)
        self._transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset"""
        return len(self._image_names)

    def __getitem__(self, index) -> tuple[Any, torch.Tensor, str]:
        """Return image, label and image name"""

        image_name = Path(self._image_names[index])

        image_path = self._image_dir / image_name
        label_path = self._label_dir / image_name.with_suffix(".txt")

        label = torch.from_numpy(np.loadtxt(fname=label_path, dtype=np.float32))

        image = Image.open(image_path).convert("RGB")

        if self._transform is not None:
            image = self._transform(image)
        # add random augmentation here
        filename = image_name.stem
        return image, label, filename


def yolic_net(weights=None) -> nn.Module:
    """Returns a Yolic model"""

    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(  # Swap out the classifier
        nn.Dropout(p=0.2),
        nn.Linear(
            in_features=model.last_channel,
            out_features=(NUMBER_OF_COIS * (NUMBER_OF_CLASSES + 1)),
        ),
        # nn.Sigmoid(),  # Should i use it here or durning the evaluation?
    )
    return model
