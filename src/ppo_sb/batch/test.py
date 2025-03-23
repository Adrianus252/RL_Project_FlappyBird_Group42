import argparse
import gym
import flappy_bird_gym
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True)
    args = parser.parse_args()

    env = gym.make("FlappyBird-v0")
    model = PPO.load(args.model_file)

    episodes = 5
    for ep in range(episodes):
        obs = env.reset()
        done = False
        reward_sum = 0
        pipe_score = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            if "score" in info:
                pipe_score = info["score"]

        print(f"Episode {ep+1}: Total Reward = {reward_sum}, Pipes = {pipe_score}")

    env.close()

if __name__ == "__main__":
    main()
