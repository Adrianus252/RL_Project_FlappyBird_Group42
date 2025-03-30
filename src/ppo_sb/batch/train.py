import argparse
import time
import csv
import gym
import flappy_bird_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# ✅ Custom training logger callback
class TrainingLogger(BaseCallback):
    def __init__(self, log_file, learning_rate, batch_size, gamma, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.episode_rewards = []
        self.episode_pipes = []
        self.episode_timesteps = []
        self.current_reward = 0
        self.current_timesteps = 0

    def _on_training_start(self):
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Total Timesteps", "Timesteps per Episode",
                "Learning Rate", "Batch Size", "Gamma", "Episodes",
                "Avg Reward", "Avg Pipes"
            ])

    def _on_step(self):
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        self.current_reward += reward
        self.current_timesteps += 1

        if done:
            pipes = info.get("score", 0)
            self.episode_rewards.append(self.current_reward)
            self.episode_pipes.append(pipes)
            self.episode_timesteps.append(self.current_timesteps)
            self.current_reward = 0
            self.current_timesteps = 0

        if self.num_timesteps % 1000 == 0:
            if self.episode_rewards:
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                avg_pipes = sum(self.episode_pipes) / len(self.episode_pipes)
                avg_timesteps = sum(self.episode_timesteps) / len(self.episode_timesteps)

                with open(self.log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"), self.num_timesteps, avg_timesteps,
                        self.learning_rate, self.batch_size, self.gamma, len(self.episode_rewards),
                        avg_reward, avg_pipes
                    ])

                self.episode_rewards.clear()
                self.episode_pipes.clear()
                self.episode_timesteps.clear()

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log_file", type=str, default="ppo_train_log.csv")
    parser.add_argument("--model_file", type=str, default="ppo_trained_model.zip")
    args = parser.parse_args()

    # Init environment
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

    # Train with logger
    logger = TrainingLogger(
        log_file=args.log_file,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma
    )
    model.learn(total_timesteps=args.timesteps, callback=logger)

    model.save(args.model_file)
    print(f"✅ Model saved to {args.model_file}")
    print(f"📊 Training log saved to {args.log_file}")


if __name__ == "__main__":
    main()
