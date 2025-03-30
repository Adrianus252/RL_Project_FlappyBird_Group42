import time
import random
import numpy as np
from stable_baselines3 import TD3
from wrapper import ContinuousFlappyBirdWrapper

# Load the trained TD3 model
model = TD3.load("trained_models/exp_16_steps200000_lr0.001_batch64_gamma0.98")

# Initialize the Flappy Bird environment
env = ContinuousFlappyBirdWrapper()

# Number of episodes to test
num_episodes = 100

# Tracking the best episode
best_reward = -float("inf")
best_seed = None  # Store the best performing seed
best_score = -1

# Function to set all random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)  # Set environment seed

# Run testing loop
for episode in range(num_episodes):
    seed = np.random.randint(0, 1000000)  # Generate a random seed
    set_seed(seed)  # Apply seed before reset
    obs = env.reset()

    episode_reward = 0
    done = False

    print(f"Running episode {episode + 1}/{num_episodes} with seed {seed}...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            print(f"Score: {info['score']}")

    # Update best episode
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_seed = seed  # Save best seed
        best_score = info['score']

print(f"\nBest episode reward: {best_reward} (Seed: {best_seed}; Score {best_score})")

# === Replay the Best Episode Using the Best Seed ===
print("\nReplaying the best episode...")
set_seed(best_seed)  # Apply the best seed before reset
obs = env.reset()  # Reset with the same seed

done = False
while not done:
    env.render()  # Render the game visually
    time.sleep(0.03)  # Adjust delay for smooth visualization
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)

print("Best episode replay finished.")
