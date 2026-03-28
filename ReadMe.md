# GAVE: Generative Auto-Bidding with Value-Guided Explorations

This is the official code for **GAVE**, accepted by SIGIR'25.

## Project Overview

GAVE is a Decision Transformer variant for auto-bidding in online advertising auctions. It uses value-guided exploration to improve bidding strategies.

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training and evaluation
python code/main/main_train_test.py
```

## Directory Structure

```
GAVE/
├── code/
│   ├── main/main_train_test.py          # Main entry point
│   ├── run/run_decision_transformer.py  # Training script
│   ├── run/run_evaluate.py              # Evaluation script
│   ├── bidding_train_env/
│   │   ├── baseline/dt/dt.py            # GAVE model
│   │   ├── environment/offline_env.py   # Bidding environment
│   │   ├── strategy/                    # Bidding strategies
│   │   └── common/                      # Utilities
│   ├── saved_model/                     # Model checkpoints
│   └── log/                             # Training logs
└── data/
    ├── trajectory/trajectory_data.csv   # Training data
    └── traffic/period-7.csv             # Test data
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--step_num` | 2 | Number of training steps |
| `--dir` | `../data/trajectory/trajectory_data.csv` | Training data path |
| `--test_csv` | `../data/traffic/period-7.csv` | Test data path |
| `--hidden_size` | 512 | Hidden layer size |
| `--batch_size` | 128 | Batch size |
| `--device` | cuda:0 | Device (cuda:0 or cpu) |
| `--expectile` | 0.99 | Expectile regression parameter |
| `--budget_rate` | 1.0 | Budget scaling factor |

## State Space (16 dimensions)

The agent observes a 16-dimensional state vector:
- `state[0]`: Time left ratio (remaining timesteps / 48)
- `state[1]`: Budget left ratio (remaining_budget / total_budget)
- `state[2-3]`: Average bid (all / last 3 timesteps)
- `state[4-5]`: Average least winning cost (all / last 3)
- `state[6-7]`: Average conversion rate (all / last 3)
- `state[8-9]`: Average xi/win rate (all / last 3)
- `state[10-11]`: Current pValue stats (avg / last 3)
- `state[12-15]`: Volume metrics (current, historical, last 3)

## Scoring Function

The evaluation uses a CPA-constrained score with penalty:

```python
def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward
```

## Loss Function

The loss function in the original paper is sensitive to hyperparameters and requires careful tuning. A more stable alternative is also provided in `code/bidding_train_env/baseline/dt/dt.py`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{gao2025generative,
  title={Generative Auto-Bidding with Value-Guided Explorations},
  author={Gao, Jingtong and Li, Yewen and Mao, Shuai and Jiang, Peng and Jiang, Nan and Wang, Yejing and Cai, Qingpeng and Pan, Fei and Gai, Kun and An, Bo and others},
  journal={arXiv preprint arXiv:2504.14587},
  year={2025}
}
```