import os
import pprint
from argparse import ArgumentParser

from typing import Dict, Any

from utils import set_seeds, init_id, init_argument_parser, init_logger


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


def main() -> None:
    arg_parser = init_argument_parser()
    arg_parser = extend_argument_parser(arg_parser)

    cfg = arg_parser.parse_args().__dict__

    set_seeds(cfg["seed"])
    if cfg["id"] is None:
        init_id(cfg)

    logger = init_logger(path_to_log=build_log_file_path(cfg))
    logger.info(pprint.pformat(cfg, indent=4))

    # TODO: add code to run task here.


if "__main__" == __name__:
    main()
