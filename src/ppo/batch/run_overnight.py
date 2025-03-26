import itertools
import subprocess
import os

# Define the Python interpreter from virtual environment
venv_python = "./Scripts/python"  # Anpassen falls nötig

# Define hyperparameter sets
episodes_list = [27500]  # Anzahl der Trainings-Episoden
#episodes_list = [50, 100, 200]  # Anzahl der Trainings-Episoden
#episodes_list = [30000]  # Anzahl der Trainings-Episoden
learning_rates = [0.0001]  # Lernrate
batch_sizes = [128,64,32]  # Batchgröße
gammas = [0.99, 0.98, 0.95]  # Discount-Faktor

# Create directories for logs and models
log_dir = "experiment_logs"
model_dir = "trained_models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Generate all hyperparameter combinations
experiments = list(itertools.product(episodes_list, learning_rates, batch_sizes, gammas))

# Run all experiments
for i, (episodes, lr, batch_size, gamma) in enumerate(experiments):
    experiment_name = f"exp_{i}_ep{episodes}_lr{lr}_batch{batch_size}_gamma{gamma}"
    log_file = f"{log_dir}/{experiment_name}.csv"
    model_file = f"{model_dir}/{experiment_name}.pth"

    print(f"\n🔹 Running Experiment {i+1}/{len(experiments)}: {experiment_name}")

    # Train the model
    train_power_command = [
        "powershell", "-Command",
        f"& py -3.9 'src/ppo/batch/train_myPPO_batch.py' "
        f"--episodes {episodes} "
        f"--learning_rate {lr} "
        f"--batch_size {batch_size} "
        f"--gamma {gamma} "
        f"--log_file '{log_file}' "
        f"--model_file '{model_file}'"
    ]
    subprocess.run(train_power_command)
    #subprocess.run("./Scripts/activate", shell=True)

print("\n✅ All experiments completed! Logs and models saved.")
