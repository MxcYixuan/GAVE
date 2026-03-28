# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GAVE (Generative Auto-Bidding with Value-Guided Explorations) is a reinforcement learning project for online advertising bidding, accepted by SIGIR'25. The project implements a Decision Transformer variant for auto-bidding in advertising auctions.

## Running the Project

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training and testing
python code/main/main_train_test.py
```

The main entry point runs both training and evaluation sequentially. You can also run them separately:

```bash
# Training only
python code/run/run_decision_transformer.py

# Evaluation only
python code/run/run_evaluate.py
```

## Data Requirements

The project requires two datasets that must be added manually (too large to include):
- Training data: `data/trajectory/trajectory_data.csv`
- Test data: `data/traffic/period-x.csv` (e.g., period-7.csv)

## Key Components

- **Model**: `code/bidding_train_env/baseline/dt/dt.py` - GAVE class implements the Decision Transformer variant with value-guided exploration
- **Training**: `code/run/run_decision_transformer.py` - trains the model using episode replay buffer
- **Evaluation**: `code/run/run_evaluate.py` - tests the model on traffic data
- **Environment**: `code/bidding_train_env/environment/offline_env.py` - simulates ad bidding
- **Strategy**: `code/bidding_train_env/strategy/dt_bidding_strategy.py` - uses trained model for bidding decisions

## Important Notes

- The loss function in `dt.py` is sensitive to hyperparameters. The default uses expectile regression with learnable value function
- There is a more stable alternative loss (commented out in dt.py) that doesn't require learnable value functions
- Model checkpoints are saved to `saved_model/` directory
- Logs are saved to `log/` directory