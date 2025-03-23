import argparse
import time
import csv
import gym
import flappy_bird_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# Custom training logger
class TrainingLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, step, episodes, avg_reward, avg_pipes, lr, batch_size, gamma):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), step, episodes, avg_reward, avg_pipes, lr, batch_size, gamma
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log_file", type=str, default="ppo_train_log.csv")
    parser.add_argument("--model_file", type=str, default="ppo_trained_model.zip")
    args = parser.parse_args()

    # Environment setup
    env = DummyVecEnv([lambda: gym.make("FlappyBird-v0")])

    # PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=1
    )

    # Train model
    model.learn(total_timesteps=args.timesteps)

    # Save model
    model.save(args.model_file)
    print(f" Model saved to {args.model_file}")


if __name__ == "__main__":
    main()
