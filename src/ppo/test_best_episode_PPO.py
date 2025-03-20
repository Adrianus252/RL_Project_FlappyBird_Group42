import time
import random
import numpy as np
import torch
import gym
import flappy_bird_gym
from my_ppo import PPOAgent  # Stelle sicher, dass dein PPOAgent importiert wird

#model 25_000_ppo_flappy.pth
#Seed: 211417; Score 36
#Seed: 845882; Score 49
#Seed: 400627; Score 85


# === Initialisiere die Umgebung ===
env = gym.make("FlappyBird-v0") # Ersetze dies mit der tatsächlichen Umgebung
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n  # Diskrete Aktionen

# === Lade den PPO-Agenten ===
agent = PPOAgent(obs_dim, act_dim)
agent.load("25_000_ppo_flappy.pth")  # Lade dein trainiertes Modell

num_episodes = 1  # Anzahl der Testepisoden

best_reward = -float("inf")
best_seed = None
best_score = -1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

# === Testschleife ===
for episode in range(num_episodes):
    seed = np.random.randint(0, 1000000)
    set_seed(seed)
    obs = env.reset()
    episode_reward = 0
    done = False
    print(f"Running episode {episode + 1}/{num_episodes} with seed {seed}...")

    while not done:
        action, log_prob, value = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            print(f"Score: {info['score']}")

    if episode_reward > best_reward:
        best_reward = episode_reward
        best_seed = seed
        best_score = info['score']

print(f"\nBest episode reward: {best_reward} (Seed: {best_seed}; Score {best_score})")

# === Wiederholung der besten Episode ===
print("\nReplaying the best episode...")
#set_seed(best_seed)
set_seed(400627)
obs = env.reset()
done = False

while not done:
    env.render()
    time.sleep(0.003)
    action, _, _ = agent.select_action(obs)
    obs, _, done, _ = env.step(action)

print("Best episode replay finished.")