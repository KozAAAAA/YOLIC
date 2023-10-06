from yolic_common import (
    NUMBER_OF_COIS,
    NUMBER_OF_CLASSES,
    YOLIC_MODEL_PATH,
    YOLIC_NET_INPUT_HEIGHT,
    YOLIC_NET_INPUT_WIDTH,
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    LABEL_DIR,
    yolic_net,
    YolicDataset,
)

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from alive_progress import alive_bar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms

EPOCHS = 150
BATCH_SIZE = 55
LEARNING_RATE = 0.001
CSV_FILE = Path("training_data.csv")


# This function has been copied from the original yolic repository,
# not fully understood yet
def yolic_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Calculate accuracy of the model"""
    accuracy = 0.0
    for i, pred in enumerate(preds):
        target = targets[i]
        target = torch.Tensor.cpu(target)
        pred = torch.Tensor.cpu(pred)

        pred = torch.round(pred).detach().numpy().astype(np.int64)
        target = target.detach().numpy()
        pred = np.reshape(pred, (NUMBER_OF_COIS * (NUMBER_OF_CLASSES + 1), 1)).flatten()
        target = np.reshape(
            target, (NUMBER_OF_COIS * (NUMBER_OF_CLASSES + 1), 1)
        ).flatten()
        num = 0
        single_accuracy = 0.0
        for cell in range(
            0, (NUMBER_OF_COIS * (NUMBER_OF_CLASSES + 1)), NUMBER_OF_CLASSES + 1
        ):
            if (
                target[cell : cell + NUMBER_OF_CLASSES + 1]
                == pred[cell : cell + NUMBER_OF_CLASSES + 1]
            ).all():
                num = num + 1

            single_accuracy = num / NUMBER_OF_COIS
        accuracy += single_accuracy
    accuracy = accuracy / len(preds)
    return accuracy


def train_step(
    model: nn.Module,
    loss_fn: Any,
    accuracy_fn: Any,
    optimizer: optim.Optimizer,
    device: torch.device,
    data_loader: DataLoader,
) -> tuple[float, float]:
    """Perform a single training step"""

    train_loss = 0.0
    train_accuracy = 0.0
    model.train()
    with alive_bar(len(data_loader)) as bar:
        for images, labels, _ in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward pass
            output = model(images)
            loss = loss_fn(output, labels)
            train_loss += loss.item()

            output = torch.sigmoid(output)  # should i use it here or in the model?
            train_accuracy += accuracy_fn(output, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            # update the progress bar
            bar()

    train_loss = train_loss / len(data_loader)
    train_accuracy = train_accuracy / len(data_loader)
    return train_loss, train_accuracy


def validation_step(
    model: nn.Module,
    loss_fn: Any,
    accuracy_fn: Any,
    device: torch.device,
    data_loader: DataLoader,
) -> tuple[float, float]:
    """Perform validation of a model"""

    validation_loss = 0.0
    validation_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = loss_fn(output, labels)
            validation_loss += loss.item()

            output = torch.sigmoid(output)  # should i use it here or in the model?
            validation_accuracy += accuracy_fn(output, labels)

    validation_loss = validation_loss / len(data_loader)
    validation_accuracy = validation_accuracy / len(data_loader)
    return validation_loss, validation_accuracy


def main():
    """Main function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Declare model, loss function, optimizer and scheduler
    model = yolic_net(weights=MobileNet_V2_Weights.DEFAULT)
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=[100, 125], gamma=0.1)

    # Declare transforms
    train_transform = transforms.Compose(
        (
            [
                transforms.Resize((YOLIC_NET_INPUT_HEIGHT, YOLIC_NET_INPUT_WIDTH)),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),
                transforms.ToTensor(),
            ]
        )
    )
    validation_transform = transforms.Compose(
        (
            [
                transforms.Resize((YOLIC_NET_INPUT_HEIGHT, YOLIC_NET_INPUT_WIDTH)),
                transforms.ToTensor(),
            ]
        )
    )

    # Create datasets
    train_dataset = YolicDataset(TRAIN_IMAGE_DIR, LABEL_DIR, train_transform)
    validation_dataset = YolicDataset(VAL_IMAGE_DIR, LABEL_DIR, validation_transform)
    print(f"Train data: {len(train_dataset)}")
    print(f"Validation data: {len(validation_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )

    # Train model
    train_loss = []
    validation_loss = []
    train_accuracy = []
    validation_accuracy = []
    previous_validation_accuracy = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch: {epoch+1} / {EPOCHS}")
        epoch_train_loss, epoch_train_accuracy = train_step(
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=yolic_accuracy,
            optimizer=optimizer,
            device=device,
            data_loader=train_loader,
        )
        epoch_validation_loss, epoch_validation_accuracy = validation_step(
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=yolic_accuracy,
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

        # TODO: Change metric to precision, recall or f1 score
        if epoch_validation_accuracy > previous_validation_accuracy:
            torch.save(model.state_dict(), YOLIC_MODEL_PATH)
            previous_validation_accuracy = epoch_validation_accuracy
            print(f"Saved new weights as {YOLIC_MODEL_PATH}")

        train_loss.append(epoch_train_loss)
        validation_loss.append(epoch_validation_loss)
        train_accuracy.append(epoch_train_accuracy)
        validation_accuracy.append(epoch_validation_accuracy)

        pd.DataFrame(
            {
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "tain_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            }
        ).to_csv(CSV_FILE, index=False)
        print(f"Added new metrics to {CSV_FILE}")


if __name__ == "__main__":
    main()
