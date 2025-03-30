import argparse
import csv
import gym
import flappy_bird_gym
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--log_file", type=str, required=True, help="Path to CSV log file for test results")
    args = parser.parse_args()

    model = PPO.load(args.model_file)
    env = gym.make("FlappyBird-v0")

    num_episodes = 100

    # Prepare CSV
    file_exists = False
    try:
        with open(args.log_file, "r") as f:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(args.log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["episode", "total_reward", "pipes_passed"])

        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            pipes_passed = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if "score" in info:
                    pipes_passed = info["score"]

            writer.writerow([ep + 1, total_reward, pipes_passed])
            print(f"Episode {ep+1}: Total Reward = {total_reward}, Pipes = {pipes_passed}")

    env.close()

if __name__ == "__main__":
    main()
