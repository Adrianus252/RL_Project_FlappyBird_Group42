import csv
import argparse
from stable_baselines3 import TD3  
from wrapper import ContinuousFlappyBirdWrapper  

# Parse command-line arguments for model and log file paths
parser = argparse.ArgumentParser()
parser.add_argument("--model_file", type=str, required=True, help="Path to the trained model file")
parser.add_argument("--log_file", type=str, required=True, help="Path to the CSV file for logging test results")
args = parser.parse_args()

# Load trained model
model = TD3.load(args.model_file)

# Initialize environment
env = ContinuousFlappyBirdWrapper()

# CSV logging
csv_file = args.log_file
file_exists = False
try:
    with open(csv_file, 'r') as f:
        file_exists = True
except FileNotFoundError:
    pass

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["episode", "total_reward", "pipes_passed"])

    for episode in range(100):  # Test for 100 episodes
        obs = env.reset()
        done = False
        total_reward = 0
        pipes_passed = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward
            if 'score' in info and done:
                pipes_passed = info['score']  # Track number of pipes passed

        # Write results to CSV
        writer.writerow([episode + 1, total_reward, pipes_passed])
        print(f"Episode {episode + 1}: Reward={total_reward}, Pipes={pipes_passed}")

env.close()
