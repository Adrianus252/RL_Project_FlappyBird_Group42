import os
import re
import numpy as np
import pandas as pd
import gym
import flappy_bird_gym

from my_ppo import PPOAgent
# Directory containing trained PPO models
MODEL_DIR = "trained_models/"
NUM_EPISODES = 100
MAX_STEPS = 1000
SEED_RANGE = 1000000
OUTPUT_CSV = "evaluation_results_ppo.csv"

# Regex pattern to extract hyperparameters from filenames
HYPERPARAM_PATTERN = re.compile(
    r"exp_\d+_ep(?P<steps>\d+)_lr(?P<lr>[\d.e-]+)_batch(?P<batch>\d+)_gamma(?P<gamma>[\d.e-]+)\.pth"
)

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

def evaluate_model(model, env, num_episodes=NUM_EPISODES):
    reward_history = []
    passed_pipes_history = []
    action_variances = []
    steps_per_episode = []

    for episode in range(num_episodes):
        seed = np.random.randint(0, SEED_RANGE)
        env.seed(seed)
        obs = env.reset()

        episode_reward = 0
        episode_pipes = 0
        done = False
        episode_actions = []
        steps = 0

        while not done:
            action, _ , _ = model.select_action(obs)
            obs, reward, done, info = env.step(action)

            episode_reward += reward
            episode_actions.append(action)
            steps += 1

            episode_pipes = info["score"]

        reward_history.append(episode_reward)
        passed_pipes_history.append(episode_pipes)
        action_variances.append(np.var(episode_actions))
        steps_per_episode.append(steps)

    avg_reward = np.mean(reward_history)
    avg_passed_pipes = np.mean(passed_pipes_history)
    convergence_speed = np.mean(pd.Series(reward_history).rolling(window=5).mean())
    stability = np.std(reward_history)
    sample_efficiency = avg_reward / np.mean(steps_per_episode)
    exploration_exploitation = np.mean(action_variances)

    return {
        "Average Reward": avg_reward,
        "Convergence Speed": convergence_speed,
        "Stability": stability,
        "Sample Efficiency": sample_efficiency,
        "Exploration vs. Exploitation": exploration_exploitation,
        "Average Passed Pipes": avg_passed_pipes
    }

def main():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    if not models:
        print("No models found in directory:", MODEL_DIR)
        return

    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    results = []

    for model_file in models:
        model_path = os.path.join(MODEL_DIR, model_file)
        print(f"Evaluating PPO model: {model_file}")

        model = PPOAgent(obs_dim, act_dim)

        # Lade die Gewichte des Modells
        model.load(model_path)

        hyperparams = extract_hyperparameters(model_file)
        metrics = evaluate_model(model, env)
        metrics.update(hyperparams)
        metrics["Model"] = model_file
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nEvaluation results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()