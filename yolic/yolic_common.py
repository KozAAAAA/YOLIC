from PIL import Image

import torch
import torch.nn as nn

import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v2

NUMBER_OF_COIS = 104
NUMBER_OF_CLASSES = 11

IMAGE_DIR = Path("./data/images/")
LABEL_DIR = Path("./data/labels/")

YOLIC_MODEL_PATH = Path("yolic_model.pt")


# TODO: lacks data augmentation as in the original code
# TODO: if there is no transform applied, image is returned as PIL.Image
class YolicDataset(Dataset):
    """Yolic Dataset class used to load images and label"""

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        image_names: list[str],
        transform=None,
    ):
        """Initialize YolicDataset class"""
        self._image_dir = image_dir
        self._label_dir = label_dir
        self._image_names = image_names
        self._transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset"""
        return len(self._image_names)

    def __getitem__(
        self, index
    ) -> tuple[Image.Image | torch.Tensor, torch.Tensor, str]:
        """Return image, label and image name"""

        image_name = self._image_names[index]

        image_path = self._image_dir / image_name
        label_path = self._label_dir / Path(image_name).with_suffix(".txt")

        label = np.loadtxt(label_path, dtype=np.float32)
        label = torch.from_numpy(label)

        image = Image.open(image_path).convert("RGB")

        if self._transform is not None:
            image = self._transform(image)
        # add random augmentation here
        return image, label, image_name


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
