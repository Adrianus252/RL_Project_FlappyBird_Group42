import time
import random
import numpy as np
import torch
import gym
import flappy_bird_gym
from my_ppo import PPOAgent

# === Initialisiere die Umgebung ===
env = gym.make("FlappyBird-v0") # Ersetze dies mit der tatsächlichen Umgebung
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n  # Diskrete Aktionen

# === Lade den PPO-Agenten ===
agent = PPOAgent(obs_dim, act_dim)
#agent.load("ppo_flappy.pth")  # Lade dein trainiertes Modell
agent.load("trained_models\exp_0_ep26000_lr3e-05_batch128_gamma0.9999.pth")
#agent.load("trained_models\exp_72_ep27500_lr0.0001_batch32_gamma0.95.pth")
#agent.load("EXCEPT_ppo_flappy.pth")
num_episodes = 100  # Anzahl der Testepisoden

best_reward = -float("inf")
best_seed = None
best_score = -1
score_array = []

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
    print(f"Running episode {episode + 1}/{num_episodes} with seed {seed}... best score: {best_score}    best seed: {best_seed}")

    while not done:
        action, log_prob, value = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            print(f"Score: {info['score']}")
            score_array.append(info['score'])
    #score_array.append(info['score'])

    if episode_reward > best_reward:
        best_reward = episode_reward
        best_seed = seed
        best_score = info['score']

print(f"\nBest episode reward: {best_reward} (Seed: {best_seed}; Score {best_score})")
print("AVG_Score: ", (sum(score_array) / num_episodes) )
print(score_array)
#26.000
#(Seed: 781802; Score 211)
#(Seed: 845647; Score 216)
#(Seed: 569581; Score 227)

# === Wiederholung der besten Episode ===
print("\nReplaying the best episode...")
#set_seed(best_seed)
set_seed(569581)
obs = env.reset()
done = False

while not done:
    env.render()
    time.sleep(0.0075)
    action, _, _ = agent.select_action(obs)
    obs, _, done, _ = env.step(action)

print("Best episode replay finished.")