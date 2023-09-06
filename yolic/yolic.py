import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader

from torchvision.models import mobilenet_v2, MobileNetV2, MobileNet_V2_Weights

from sklearn.model_selection import train_test_split

EPOCHS = 150
BATCH_SIZE = 32

LEARNING_RATE = 0.001
NUMBER_OF_COI = 104
NUMBER_OF_CLASSES = 11

DATA_RATIO_TRAIN = 0.6
DATA_RATIO_VALIDATION = 0.2
DATA_RATIO_TEST = 0.2

IMAGE_DIR = "./data/images/"
LABEL_DIR = "./data/labels/"


def YolicDataset(Dataset):
    """Yolic Dataset class used to load images and label"""

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        data_filenames: list[str],
        transform=None,
    ):
        self._image_dir = image_dir
        self._label_dir = label_dir
        self._data_filenames = data_filenames
        self.transform = transform

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def main():
    """Main function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(
        in_features=model.last_channel,
        out_features=(NUMBER_OF_COI * (NUMBER_OF_CLASSES + 1)),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=[100, 125], gamma=0.1)


if __name__ == "__main__":
    main()
