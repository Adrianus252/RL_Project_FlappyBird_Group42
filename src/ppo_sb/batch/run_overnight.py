import itertools
import subprocess
import os
import time


# Define the path to the virtual environment's Python interpreter
# Modify this to the correct path of your venv's python executable
venv_python = "./venv/bin/python"  


# Define hyperparameter sets
timesteps_list = [500000]  # Number of timesteps for training
learning_rates = [0.0001, 0.001, 0.005, 0.01]  # Learning rate for optimizer
batch_sizes = [32, 64, 128, 256]  # Batch size for updates
gammas = [0.95, 0.98, 0.99]  # Discount factor (gamma)


# Create folder to store logs & models
log_dir = "experiment_logs"
model_dir = "trained_models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Generate all hyperparameter combinations
experiments = list(itertools.product(timesteps_list, learning_rates, batch_sizes, gammas))

# Run all experiments
for i, (timesteps, lr, batch_size, gamma) in enumerate(experiments):
    experiment_name = f"exp_{i}_steps{timesteps}_lr{lr}_batch{batch_size}_gamma{gamma}"
    log_file = f"{log_dir}/{experiment_name}.csv"
    model_file = f"{model_dir}/{experiment_name}.zip"

    print(f"\n🔹 Running Experiment {i+1}/{len(experiments)}: {experiment_name}")

    # Train the model
    train_command = [
        venv_python, "./src/ppo_sb/batch/train.py",
        "--timesteps", str(timesteps),
        "--learning_rate", str(lr),
        "--batch_size", str(batch_size),
        "--gamma", str(gamma),
        "--log_file", log_file,
        "--model_file", model_file
    ]
    # train_command = [
    #     venv_python, "./src/td3/batch/train.py",
    #     "--timesteps", str(timesteps),
    #     "--learning_rate", str(lr),
    #     "--batch_size", str(batch_size),
    #     "--gamma", str(gamma),
    #     "--log_file", log_file,
    #     "--model_file", model_file
    # ]
    subprocess.run(train_command)

    # Test the model
    test_command = [
        venv_python, "./src/ppo_sb/batch/test.py",
        "--model_file", model_file,
        "--log_file", log_file  # Append test results to same file
    ]
    subprocess.run(test_command)

print("\n✅ All experiments completed! Logs and models saved.")
