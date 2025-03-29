import itertools
import subprocess
import os
import time

# Define the path to the virtual environment's Python interpreter
# Modify this to the correct path of your venv's python executable
venv_python = "./venv/bin/python"  

# Define the specific hyperparameter combinations you want
experiments = [
    (200000, 0.001, 64, 0.98),
    (500000, 0.0001, 32, 0.98),
    (500000, 0.001, 32, 0.95),
    (500000, 0.001, 128, 0.95),
    (100000, 0.001, 256, 0.95)
]

# Create folder to store logs & models
log_dir = "experiment_logs"
model_dir = "trained_models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Run all experiments
for i, (timesteps, lr, batch_size, gamma) in enumerate(experiments):
    experiment_name = f"exp_{i+1}_steps{timesteps}_lr{lr}_batch{batch_size}_gamma{gamma}"
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
    subprocess.run(train_command)

    # Test the model
    test_command = [
        venv_python, "./src/ppo_sb/batch/test.py",
        "--model_file", model_file,
        "--log_file", log_file  # Append test results to same file
    ]
    subprocess.run(test_command)

print("\n✅ All experiments completed! Logs and models saved.")
