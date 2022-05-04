import os
import pprint
from argparse import ArgumentParser

from typing import Dict, Any

from sklearn.metrics import accuracy_score

import torch
from torch import nn
import torch.nn.functional as F

from utils import set_seeds, init_id, init_argument_parser, init_logger
from data import get_img_dataset


def extend_argument_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add task-specific CLI arguments to a basic CLI argument parser."""
    # TODO: Add task-specific CLI arguments here.
    return parser


def build_log_file_path(cfg: Dict[str, Any]) -> str:
    """Construct path to log file from configuration."""
    # TODO: Customize this if needed.
    path = os.path.join(
        cfg["checkpoints"],
        "task2",
    )
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"experiment-{cfg['id']}.log")
    return file_path


class BaselineClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                3, 16, kernel_size=3, stride=2, padding=1, bias=False
            ),  # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                16, 32, kernel_size=3, stride=2, padding=1, bias=False
            ),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1, bias=False
            ),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1, bias=False
            ),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                128, 256, kernel_size=3, stride=2, padding=1, bias=False
            ),  # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                256, 512, kernel_size=3, stride=2, padding=1, bias=False
            ),  # 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels = batch
        # Get predictions
        out = self(images)
        # Get loss
        loss = F.cross_entropy(out, labels)
        return loss

    @torch.no_grad()
    def validation_step(self, batch):
        images, labels = batch
        # Get predictions
        out = self(images)
        # Get loss
        loss = F.cross_entropy(out, labels)
        # Get accuracy
        _, preds = torch.max(out, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        return {"val_loss": loss, "val_acc": acc}


def main() -> None:
    arg_parser = init_argument_parser()
    arg_parser = extend_argument_parser(arg_parser)

    cfg = arg_parser.parse_args().__dict__

    set_seeds(cfg["seed"])
    if cfg["id"] is None:
        init_id(cfg)

    logger = init_logger(path_to_log=build_log_file_path(cfg))
    logger.info(pprint.pformat(cfg, indent=4))

    logger.info("Loading MRI image dataset...")
    data_dir = os.path.join(cfg["data"], "images")
    train, valid, test = get_img_dataset(data_dir=data_dir, logger=logger)
    logger.info("MRI image dataset loaded!")

    # TODO: add code to run task here, current code is just dummy for testing.
    model = BaselineClf()

    logger.info("First train sample: %s", train.dataset[0])
    logger.info("First valid sample: %s", valid.dataset[0])
    logger.info("First test sample: %s", test.dataset[0])

    logger.info("Baseline model: %s", model)


if "__main__" == __name__:
    main()
