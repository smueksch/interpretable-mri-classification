import os
import pprint
from argparse import ArgumentParser
import logging
import time

from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)

from utils import set_seeds, init_id, init_argument_parser, init_logger
from data import get_img_data_loaders


def extend_argument_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add task-specific CLI arguments to a basic CLI argument parser."""
    # TODO: Add task-specific CLI arguments here.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used in train, val and test data loaders.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers used in train, val and test data loaders.",
    )
    parser.add_argument(
        "--gradient_max_norm", type=float, default=1.0, help="Maximum gradient norm.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=20, help="Maximum number of epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for optimizer."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--lr_scheduler_patience",
        type=int,
        default=3,
        help="Maximum number of epochs without an improvement on validation loss before the learning rate is reduced.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Maximum number of epochs without an improvement on validation loss before the training is terminated.",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        type=bool,
        default=True,
        help="Whether to use a learning rate scheduler or not.",
    )
    parser.add_argument(
        "--model_name", type=str, default="baseline_cnn", help="Model name."
    )
    return parser


def train_epoch(
    cfg: Dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data_loader: torch.utils.data.DataLoader,
    class_weights: torch.Tensor,
) -> Dict[str, float]:
    model.train()
    all_y = []
    all_yhat = []
    for batch in train_data_loader:
        optimizer.zero_grad()
        X, y = batch
        X = X.float().to(cfg["device"])
        y = y.to(cfg["device"])
        yhat = model(X)
        sample_weights = torch.tensor(
            [class_weights[int(label)] for label in y],
            dtype=torch.float,
            device=cfg["device"],
        )
        cross_entropy_loss = torch.nn.BCEWithLogitsLoss(weight=sample_weights)(
            yhat.squeeze(), y.float()
        )
        all_y.append(y.detach().cpu().numpy())
        all_yhat.append(yhat.detach().cpu().numpy())
        cross_entropy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_max_norm"])
        optimizer.step()
    all_y = np.concatenate(all_y, axis=0)
    all_yhat = np.concatenate(all_yhat, axis=0).astype(np.float32)
    train_loss_dict = evaluate_predictions(cfg, all_y, all_yhat, class_weights)
    return train_loss_dict


def evaluation_epoch(
    cfg: Dict[str, Any],
    model: nn.Module,
    evaluation_data_loader: torch.utils.data.DataLoader,
    class_weights: torch.Tensor,
    split: str,
    save_to_disk: bool = False,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        all_y = []
        all_yhat = []
        for batch in evaluation_data_loader:
            X, y = batch
            X = X.float().to(cfg["device"])
            y = y.to(cfg["device"])
            yhat = model(X)
            all_y.append(y.detach().cpu().numpy())
            all_yhat.append(yhat.detach().cpu().numpy())
        all_y = np.concatenate(all_y, axis=0)
        all_yhat = np.concatenate(all_yhat, axis=0).astype(np.float32)
        if save_to_disk:
            save_predictions_to_disk(cfg, all_y, all_yhat, split)
        eval_loss_dict = evaluate_predictions(cfg, all_y, all_yhat, class_weights)
    return eval_loss_dict


def evaluate_predictions(
    cfg: Dict[str, Any],
    all_y: torch.Tensor,
    all_yhat: torch.Tensor,
    class_weights: torch.Tensor,
) -> Dict[str, float]:
    sample_weights = torch.tensor(
        [class_weights[int(label)] for label in all_y],
        dtype=torch.float,
        device=cfg["device"],
    )
    result_dict = {}
    all_yhat_probs = expit(all_yhat)
    result_dict["cross_entropy_loss"] = float(
        torch.nn.BCEWithLogitsLoss(weight=sample_weights)(
            torch.tensor(all_yhat, device=cfg["device"], dtype=torch.float).squeeze(),
            torch.tensor(all_y, device=cfg["device"], dtype=torch.float),
        )
    )
    all_yhat_argmaxed = 1 * (all_yhat_probs >= 0.5)
    result_dict["unbalanced_acc_score"] = accuracy_score(all_y, all_yhat_argmaxed)
    result_dict["balanced_acc_score"] = balanced_accuracy_score(
        all_y, all_yhat_argmaxed
    )
    result_dict["roc_auc_score"] = roc_auc_score(all_y, all_yhat_probs)
    result_dict["pr_auc_score"] = average_precision_score(all_y, all_yhat_probs)
    return result_dict


def get_checkpoints_dir(cfg: Dict[str, Any]) -> str:
    model_name = cfg["model_name"]
    checkpoints_dir = os.path.join(
        cfg["checkpoints"], "task2", cfg["experiment_time"] + "_" + cfg["id"]
    )
    os.makedirs(checkpoints_dir, exist_ok=True)
    return checkpoints_dir


def save_predictions_to_disk(
    cfg: Dict[str, Any], all_y: torch.Tensor, all_yhat: torch.Tensor, split: str
):
    checkpoints_dir = get_checkpoints_dir(cfg)
    predictions_path = os.path.join(checkpoints_dir, f"{split}_predictions.txt")

    logit_1 = all_yhat
    prob_1 = expit(logit_1)
    prob_0 = 1 - prob_1
    all_yhat_probs = np.hstack((prob_0, prob_1))
    columns = ["prob_0", "prob_1", "label"]
    df = pd.DataFrame(
        np.hstack((all_yhat_probs, all_y.reshape(-1, 1))), columns=columns
    )
    df.to_csv(predictions_path, index=False)


class BaselineCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),  # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Produces logit for class 0
            # nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Inputs:
                x: Input images. A torch.Tensor with shape (cfg["batch_size"], 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            Output:
                yhat: Logits for class 0. A torch.Tensor with shape (cfg["batch_size"], 1)
        """
        return self.model(x)


def get_model(cfg: Dict[str, Any]) -> nn.Module:
    if cfg["model_name"] == "baseline_cnn":
        return BaselineCNN(cfg).to(cfg["device"])
    else:
        raise Exception(f"Not a valid model {model}.")


def get_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )


def get_scheduler(
    cfg: Dict[str, Any], optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg["lr_scheduler_patience"], verbose=True
    )


def save_checkpoint(
    cfg: Dict[str, Any], model: nn.Module, logger: logging.Logger
) -> None:
    logger.info("Saving the best checkpoint...", cfg)
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_dict = {"model_state_dict": model.state_dict()}
    torch.save(checkpoint_dict, os.path.join(checkpoints_dir, "best_checkpoint"))


def load_checkpoint(cfg: Dict[str, Any], logger: logging.Logger) -> nn.Module:
    logger.info("Loading the best checkpoint...", cfg)
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_dict = torch.load(os.path.join(checkpoints_dir, "best_checkpoint"))
    model = get_model(cfg)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    return model


def main() -> None:
    arg_parser = init_argument_parser()
    arg_parser = extend_argument_parser(arg_parser)

    cfg = arg_parser.parse_args().__dict__
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["experiment_time"] = str(int(time.time()))

    set_seeds(cfg["seed"])
    if cfg["id"] is None:
        init_id(cfg)

    checkpoints_dir = get_checkpoints_dir(cfg)
    log_file_path = os.path.join(checkpoints_dir, "logs.txt")

    logger = init_logger(path_to_log=log_file_path, level=logging.INFO)
    logger.info(pprint.pformat(cfg, indent=4))

    logger.info("Loading MRI image dataset...")
    data_dir = os.path.join(cfg["data"], "images")

    # TODO: Add some additional train transforms
    (
        train_data_loader,
        val_data_loader,
        test_data_loader,
        class_weights,
    ) = get_img_data_loaders(
        cfg=cfg,
        data_dir=data_dir,
        logger=logger,
        additional_train_transforms=None,
        shuffle_train=True,
    )
    logger.info("MRI image dataset loaded!")

    model = get_model(cfg=cfg)
    optimizer = get_optimizer(cfg=cfg, model=model)

    if cfg["use_lr_scheduler"]:
        scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)

    best_val_loss = np.inf
    early_stop_counter = 0
    for epoch in range(cfg["max_epochs"]):
        train_loss_dict = train_epoch(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            class_weights=class_weights,
        )
        logger.info(
            f"Train | Epoch: {epoch+1}, "
            + ", ".join(
                [
                    f"{loss_function}: {np.round(loss_value, 3)}"
                    for loss_function, loss_value in train_loss_dict.items()
                ]
            )
        )

        val_loss_dict = evaluation_epoch(
            cfg=cfg,
            model=model,
            evaluation_data_loader=val_data_loader,
            class_weights=class_weights,
            split="val",
            save_to_disk=False,
        )
        current_val_loss = val_loss_dict["cross_entropy_loss"]

        logger.info(
            f"Validation | Epoch: {epoch+1}, "
            + ", ".join(
                [
                    f"{loss_function}: {np.round(loss_value, 3)}"
                    for loss_function, loss_value in val_loss_dict.items()
                ]
            )
        )

        if cfg["use_lr_scheduler"]:
            scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss:
            save_checkpoint(cfg=cfg, model=model, logger=logger)
            best_val_loss = current_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter == cfg["early_stop_patience"]:
            break

    model = load_checkpoint(cfg=cfg, logger=logger)

    (
        train_data_loader,
        val_data_loader,
        test_data_loader,
        class_weights,
    ) = get_img_data_loaders(
        cfg=cfg,
        data_dir=data_dir,
        logger=logger,
        additional_train_transforms=None,
        shuffle_train=False,
    )

    evaluation_epoch(
        cfg=cfg,
        model=model,
        evaluation_data_loader=train_data_loader,
        class_weights=class_weights,
        split="train",
        save_to_disk=True,
    )
    evaluation_epoch(
        cfg=cfg,
        model=model,
        evaluation_data_loader=val_data_loader,
        class_weights=class_weights,
        split="val",
        save_to_disk=True,
    )
    test_loss_dict = evaluation_epoch(
        cfg=cfg,
        model=model,
        evaluation_data_loader=test_data_loader,
        class_weights=class_weights,
        split="test",
        save_to_disk=True,
    )

    logger.info(
        f"Test | "
        + ", ".join(
            [
                f"{loss_function}: {np.round(loss_value, 3)}"
                for loss_function, loss_value in test_loss_dict.items()
            ]
        )
    )


if "__main__" == __name__:
    main()
