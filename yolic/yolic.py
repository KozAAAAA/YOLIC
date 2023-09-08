from os import listdir, pread

from pathlib import Path

import numpy as np

from PIL import Image
from sklearn.utils import shuffle

from alive_progress import alive_bar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms

from sklearn.model_selection import train_test_split


EPOCHS = 150
BATCH_SIZE = 55

LEARNING_RATE = 0.001
NUMBER_OF_COI = 104
NUMBER_OF_CLASSES = 11

IMAGE_DIR = Path("./data/images/")
LABEL_DIR = Path("./data/labels/")

YOLIC_MODEL_PATH = Path("yolic_model.pt")

INPUT_WIDTH = 224
INPUT_HEIGHT = 224


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

    def __len__(self):
        """Return the length of the dataset"""
        return len(self._image_names)

    def __getitem__(self, index):
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


def split_dataset(
    data: list,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
    test_ratio: float = 0.20,
):
    """Split data into train, validation and test sets"""
    assert train_ratio + validation_ratio + test_ratio == 1
    train, left_data = train_test_split(data, test_size=1 - train_ratio)
    validation, test = train_test_split(
        left_data,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=1,
    )
    return train, validation, test


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    data_loader: DataLoader,
) -> tuple[float, float]:
    """Perform a single training step"""

    train_loss = 0.0
    model.train()
    with alive_bar(len(data_loader)) as bar:
        for _, (images, labels, _) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            output = model(images)
            loss = loss_fn(output, labels)
            train_loss += loss.item()

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            bar()

    train_loss = train_loss / len(data_loader)
    train_accuracy = 0.0  # add accuracy calculation here
    return train_loss, train_accuracy


def validation_step(
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
) -> tuple[float, float]:
    """Perform validation of a model"""

    validation_loss = 0.0
    model.eval()
    with torch.no_grad():
        for _, (images, labels, _) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_fn(output, labels)
            validation_loss += loss.item()

    validation_loss = validation_loss / len(data_loader)
    validation_accuracy = 0.0  # add accuracy calculation here
    return validation_loss, validation_accuracy


def main():
    """Main function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Declare model, loss function, optimizer and scheduler
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(
        in_features=model.last_channel,
        out_features=(NUMBER_OF_COI * (NUMBER_OF_CLASSES + 1)),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=[100, 125], gamma=0.1)

    # Declare transforms
    train_transform = transforms.Compose(
        (
            [
                transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),
                transforms.ToTensor(),
            ]
        )
    )
    test_and_validation_transform = transforms.Compose(
        ([transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)), transforms.ToTensor()])
    )

    # Split datasets
    train_image_names, validation_image_names, test_image_names = split_dataset(
        listdir(IMAGE_DIR)
    )
    print(f"Train data: {len(train_image_names)}")
    print(f"Validation data: {len(validation_image_names)}")
    print(f"Test data: {len(test_image_names)}")

    # Create datasets
    train_dataset = YolicDataset(
        IMAGE_DIR, LABEL_DIR, train_image_names, train_transform
    )
    validation_dataset = YolicDataset(
        IMAGE_DIR, LABEL_DIR, validation_image_names, test_and_validation_transform
    )
    test_dataset = YolicDataset(
        IMAGE_DIR, LABEL_DIR, test_image_names, test_and_validation_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )

    # Train model
    previous_validation_loss = float("inf")
    model.to(device)
    for epoch in range(EPOCHS):
        print(f"\nEpoch: {epoch} / {EPOCHS}")
        epoch_train_loss, epoch_train_accuracy = train_step(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            data_loader=train_loader,
        )
        epoch_validation_loss, epoch_validation_accuracy = validation_step(
            model=model,
            loss_fn=loss_fn,
            device=device,
            data_loader=validation_loader,
        )
        scheduler.step()

        print(
            f"Train loss: {epoch_train_loss},",
            f"Train accuracy: {epoch_train_accuracy}",
        )
        print(
            f"Validation loss: {epoch_validation_loss},",
            f"Validation accuracy: {epoch_validation_accuracy}",
        )

        if epoch_validation_loss < previous_validation_loss:
            torch.save(model.state_dict(), YOLIC_MODEL_PATH)
            previous_validation_loss = epoch_validation_loss
            print(f"Saved new weights as {YOLIC_MODEL_PATH}")


if __name__ == "__main__":
    main()
