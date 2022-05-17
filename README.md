# Environment

In the root of this repository, run

```sh
conda env create -f environment.yml
```

# Task 1

First, activate the environment:

```sh
conda activate ml4hc_project3
```

then navigate to the `src` folder to run the remaining commands.

To train the XGBoost classifier with the best hyperparameter settings, run:

```sh
python task1.py --retrain
```

To perform the grid search used to find the best hyperparameter settings for the
XGBoost classifier, run:

```bash
python task1.py --retrain --grid_search
```

# Task 2

```bash
python task2.py
```

# Task 3

First, activate the environment:

```sh
conda activate ml4hc_project3
```

then navigate to the `src` folder to run the remaining commands.

Run the following command for the `RuleFit` experiment:

```sh
python task3_rules.py --retrain
```