import os
import pprint
from argparse import ArgumentParser

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

from utils import set_seeds, init_id, init_argument_parser, init_logger
from data import get_radiomics_dataset


def extend_argument_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add task-specific CLI arguments to a basic CLI argument parser."""
    return parser


def build_log_file_path(cfg: Dict[str, Any]) -> str:
    """Construct path to log file from configuration."""
    path = os.path.join(
        cfg["checkpoints"],
        "task1",
    )
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"experiment-{cfg['id']}.log")
    return file_path


def select_important_features(
    features: List[str], feature_importances: List[float]
) -> Dict[str, float]:
    assert len(features) == len(
        feature_importances
    ), "Need as many feature importances as features!"
    important_features = zip(features, feature_importances)
    return {t[0]: t[1] for t in important_features if t[1] > 0.0}


def combine_train_and_valid(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X_train_valid = np.concatenate((X_train.to_numpy(), X_valid.to_numpy()))
    y_train_valid = np.concatenate((y_train, y_valid))
    return X_train_valid, y_train_valid


def search_grid(
    cfg: Dict[str, Any], X_train_valid: np.ndarray, y_train_valid: np.ndarray
) -> GridSearchCV:
    parameter_grid = {
        "n_estimators": [700, 800, 900, 1000],
        "max_depth": [3, 4],
        "learning_rate": [0.01],
        # "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        # "max_depth": [3, 4, 5, 6],
        # "learning_rate": [0.01, 0.05, 0.1],
        # "gamma": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        # "subsample": [0.7, 0.8, 0.9, 1.0],
        # "reg_alpha": [1e-5, 1e-4, 1e-3],
        # "reg_lambda": [0.1, 0.01, 1],
    }

    xgbc = XGBClassifier(n_jobs=16, random_state=cfg["seed"])

    grid_search = GridSearchCV(
        xgbc,
        parameter_grid,
        scoring="accuracy",
        n_jobs=16,
        refit=True,
        cv=5,
        verbose=4,
        return_train_score=True,
    )

    grid_search.fit(X_train_valid, y_train_valid)
    return grid_search


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

    logger.info("X_train shape: %s", X_train.shape)
    logger.info("y_train shape: %s", y_train.shape)
    logger.info("X_valid shape: %s", X_valid.shape)
    logger.info("y_valid shape: %s", y_valid.shape)
    logger.info("X_test shape: %s", X_test.shape)
    logger.info("y_test shape: %s", y_test.shape)

    X_train_valid, y_train_valid = combine_train_and_valid(
        X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid
    )

    logger.info("X_train_valid shape: %s", X_train.shape)
    logger.info("y_train_valid shape: %s", y_train.shape)

    logger.info("Starting grid search for XGBoost Classifier...")
    grid_search = search_grid(cfg, X_train_valid, y_train_valid)
    logger.info("Grid search complete!")

    xgbc = grid_search.best_estimator_

    logger.info(
        "Best hyperparameter settings: %s",
        pprint.pformat(grid_search.best_params_, indent=4),
    )

    important_features = select_important_features(
        X_train.columns.to_list(), xgbc.feature_importances_.tolist()
    )

    logger.info(
        "Important features (%d/%d): %s",
        len(important_features),
        len(X_train.columns),
        pprint.pformat(important_features, indent=4),
    )

    logger.info(
        "Training set accuracy: %f",
        accuracy_score(y_true=y_train, y_pred=xgbc.predict(X_train)),
    )
    logger.info(
        "Validation set accuracy: %f",
        accuracy_score(y_true=y_valid, y_pred=xgbc.predict(X_valid)),
    )
    logger.info(
        "Test set accuracy: %f",
        accuracy_score(y_true=y_test, y_pred=xgbc.predict(X_test)),
    )


if "__main__" == __name__:
    main()
