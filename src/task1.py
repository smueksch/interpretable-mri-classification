import os
import pprint
from argparse import ArgumentParser

from typing import Dict, Any

from utils import set_seeds, init_id, init_argument_parser, init_logger
from data import get_radiomics_dataset


def extend_argument_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add task-specific CLI arguments to a basic CLI argument parser."""
    # TODO: Add task-specific CLI arguments here.
    return parser


def build_log_file_path(cfg: Dict[str, Any]) -> str:
    """Construct path to log file from configuration."""
    # TODO: Customize this if needed.
    path = os.path.join(
        cfg["checkpoints"],
        "task1",
    )
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"experiment-{cfg['id']}.log")
    return file_path


def main() -> None:
    arg_parser = init_argument_parser()
    arg_parser = extend_argument_parser(arg_parser)

    cfg = arg_parser.parse_args().__dict__

    set_seeds(cfg["seed"])
    if cfg["id"] is None:
        init_id(cfg)

    logger = init_logger(path_to_log=build_log_file_path(cfg))
    logger.info(pprint.pformat(cfg, indent=4))

    logger.info("Loading radiomics dataset...")
    data_dir = os.path.join(cfg["data"], "radiomics")
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_radiomics_dataset(
        data_dir=data_dir
    )
    logger.info("Radiomics dataset loaded!")

    # TODO: add code to run task here, current code is just dummy for testing.
    logger.info("X_train shape: %s", X_train.shape)
    logger.info("y_train shape: %s", y_train.shape)
    logger.info("X_valid shape: %s", X_valid.shape)
    logger.info("y_valid shape: %s", y_valid.shape)
    logger.info("X_test shape: %s", X_test.shape)
    logger.info("y_test shape: %s", y_test.shape)


if "__main__" == __name__:
    main()
