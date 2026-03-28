"""
GAVE (Generative Auto-Bidding with Value-Guided Explorations) - Main Entry Point
Run this file to train and test the model
"""
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Change to code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from run.run_decision_transformer import run_dt
from run.run_evaluate import run_test
import glob
import pandas as pd
import datetime

if __name__ == "__main__":
    # Get absolute paths (data is in parent directory)
    data_dir = os.path.join(os.path.dirname(project_root), "data")

    # Configuration
    config = {
        "step_num": 50,           # Training steps
        "save_step": 25,         # Save model every N steps
        "dir": os.path.join(data_dir, "trajectory/trajectory_data.csv"),
        "test_csv": os.path.join(data_dir, "traffic/period-x.csv"),
        "hidden_size": 128,       # Smaller for faster training
        "learning_rate": 0.0001,
        "time_dim": 8,
        "batch_size": 64,
        "device": "cpu",
        "expectile": 0.99,
        "loss_report": 10,
        "budget_rate": 1.0,
        "block_config": {
            "n_ctx": 256,
            "n_embd": 128,
            "n_layer": 3,
            "n_head": 4,
            "n_inner": 256,
            "activation_function": "relu",
            "n_position": 256,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
        }
    }

    print("=" * 50)
    print("GAVE - Generative Auto-Bidding")
    print("=" * 50)
    print("\nConfiguration:")
    for k, v in config.items():
        if k != "block_config":
            print(f"  {k}: {v}")

    # Create necessary directories
    log_dir = os.path.join(project_root, "log")
    model_dir = os.path.join(project_root, "saved_model")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save parameters
    time_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(log_dir, config["test_csv"].split("/")[-1].replace(".csv", "") + "_" + time_now)
    with open(filename + "_param.txt", 'w') as f:
        f.write(str(config))
    config['save_dir'] = os.path.join(project_root, "saved_model/GAVE_" + time_now)

    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)
    run_dt(
        device=config["device"],
        step_num=config["step_num"],
        dir=config["dir"],
        save_step=config["save_step"],
        model_param=config,
        batch_size=config["batch_size"],
        save_dir=config['save_dir'],
        loss_report=config['loss_report']
    )

    print("\n" + "=" * 50)
    print("Testing...")
    print("=" * 50)
    config["device"] = "cpu"
    csv_file = config["test_csv"]
    pt_files = glob.glob(os.path.join(config['save_dir'], '*.pt'))
    pt_names = sorted([int(f.split("/")[-1].replace(".pt", "")) for f in pt_files])

    eval_result = []
    for pt_name in pt_names:
        score, score1, conversion, exc = run_test(
            file_path=csv_file,
            model_name=str(pt_name) + ".pt",
            model_param=config
        )
        eval_result.append([pt_name, score, score1, conversion, exc])
        print(f"Model {pt_name}.pt -> Score: {score:.2f}, Exceed Rate: {exc:.2%}")

    # Save results
    eval_result_csv = pd.DataFrame(
        eval_result,
        columns=["file", "score", "score1", "conversion", "exceed"]
    ).sort_values(by="file")
    eval_result_csv.to_csv(filename + "_result.csv", index=False)

    print("\n" + "=" * 50)
    print("Done! Results saved to:", filename + "_result.csv")
    print("=" * 50)