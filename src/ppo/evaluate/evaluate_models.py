import os
import re
import numpy as np
import pandas as pd
import torch
#from src.my_ppo import PPOAgent  # Hier musst du deinen PPO-Agent importieren
from my_ppo import PPOAgent
import gym
import flappy_bird_gym

# Directory containing trained models
MODEL_DIR = "trained_models/"
NUM_EPISODES = 100  # Number of test episodes per model
MAX_STEPS = 1000  # Max steps per episode
SEED_RANGE = 1000000  # Range for generating random seeds
OUTPUT_CSV = "evaluation_results.csv"

# Regex pattern to extract hyperparameters from filenames
HYPERPARAM_PATTERN = re.compile(
    r"exp_\d+_ep(?P<steps>\d+)_lr(?P<lr>[\d.e-]+)_batch(?P<batch>\d+)_gamma(?P<gamma>[\d.e-]+)\.pth"
)

# Function to extract hyperparameters from filename
def extract_hyperparameters(filename):
    match = HYPERPARAM_PATTERN.search(filename)
    if match:
        return {
            "Steps": int(match.group("steps")),
            "Learning Rate": float(match.group("lr")),
            "Batch Size": int(match.group("batch")),
            "Gamma": float(match.group("gamma"))
        }
    return {"Steps": None, "Learning Rate": None, "Batch Size": None, "Gamma": None}

# Function to evaluate a model
def evaluate_model(model, env, num_episodes=NUM_EPISODES):
    reward_history = []
    action_variances = []
    steps_per_episode = []

    for episode in range(num_episodes):
        seed = np.random.randint(0, SEED_RANGE)
        env.seed(seed)
        obs = env.reset()

        episode_reward = 0
        done = False
        episode_actions = []
        steps = 0

        while not done and steps < MAX_STEPS:
            action, _, _ = model.select_action(obs)  # Verwende select_action() von PPO-Agent
            obs, reward, done, _ = env.step(action)

            episode_reward += reward
            episode_actions.append(action)
            steps += 1

        reward_history.append(episode_reward)
        action_variances.append(np.var(episode_actions))
        steps_per_episode.append(steps)

    # Compute metrics
    avg_reward = np.mean(reward_history)
    convergence_speed = np.mean(pd.Series(reward_history).rolling(window=5).mean())  # Smoothed reward
    stability = np.std(reward_history)  # Standard deviation of rewards
    sample_efficiency = avg_reward / np.mean(steps_per_episode)  # Reward per step
    exploration_exploitation = np.mean(action_variances)  # Action variance

    return {
        "Average Reward": avg_reward,
        "Convergence Speed": convergence_speed,
        "Stability": stability,
        "Sample Efficiency": sample_efficiency,
        "Exploration vs. Exploitation": exploration_exploitation
    }

# Main function to run evaluation on all models
def main():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]  # Jetzt suchen wir .pth Modelle
    if not models:
        print("No models found in directory:", MODEL_DIR)
        return

    #env = FlappyBirdEnv()  # Direkt den FlappyBird-Env laden
    # === Initialisiere die Umgebung ===
    env = gym.make("FlappyBird-v0") # Ersetze dies mit der tatsächlichen Umgebung
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # Diskrete Aktionen
    results = []

    for model_file in models:
        model_path = os.path.join(MODEL_DIR, model_file)
        print(f"Evaluating model: {model_file}")

        model = PPOAgent(obs_dim, act_dim)  # Dein PPO-Agent

        # Lade die Gewichte des Modells
        model.load(model_path)  # Jetzt lädt das PPO-Modell die gespeicherten .pth-Dateien

        # Extract hyperparameters from filename (optional)
        hyperparams = extract_hyperparameters(model_file)
        print(hyperparams)
        # Evaluate model
        metrics = evaluate_model(model, env)
        metrics.update(hyperparams)  # Merge hyperparameters into results
        metrics["Model"] = model_file  # Add model name
        results.append(metrics)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nEvaluation results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
