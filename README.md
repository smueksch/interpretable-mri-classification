# Environment

```bash
python -m venv ml4hc-p3
source ml4hc-p3/bin/activate
python -m pip install -r requirements.txt
```

> N.B.: If you are using Windows, please refer to the
> [official documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
> on how to create and activate virtual environments.

# Task 1

```bash
python task1.py
```

# Task 2

```bash
bsub -n 3 -W 24:00 -R "rusage[mem=8192]" python task2.py --model_name baseline_cnn
```

# Task 3

```bash
python task3.py
```

# Optional Task

```bash
chmod +x optional_task_commands.sh
./optional_task_commands.sh
```
