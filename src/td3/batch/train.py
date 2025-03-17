import argparse
import time
import csv
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from wrapper import ContinuousFlappyBirdWrapper

# Custom callback to log training data
class TrainingLogger(BaseCallback):
    def __init__(self, log_file="train_log.csv", learning_rate=0.001, batch_size=64, gamma=0.99):
        super().__init__()
        self.log_file = log_file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.episode_rewards = []
        self.episode_pipes = []
        self.episode_timesteps = []
        self.current_episode_reward = 0
        self.current_episode_pipes = 0
        self.current_episode_timesteps = 0

    def _on_training_start(self) -> None:
        """Initialize logging file at the start of training."""
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Total Timesteps", "Timesteps per Episode", "Learning Rate", "Batch Size", "Gamma", "Episodes", "Avg Reward", "Avg Pipes"])

    def _on_step(self) -> bool:
        """Log training progress every 1000 timesteps."""
        done = self.locals["dones"][0]
        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0] if "infos" in self.locals and self.locals["infos"] else {}

        self.current_episode_reward += reward
        self.current_episode_timesteps += 1

        if done:
            pipes_passed = info.get("score", 0)
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_pipes.append(pipes_passed)
            self.episode_timesteps.append(self.current_episode_timesteps)

            self.current_episode_reward = 0
            self.current_episode_pipes = 0
            self.current_episode_timesteps = 0

        if self.num_timesteps % 1000 == 0:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
            avg_pipes = sum(self.episode_pipes) / len(self.episode_pipes) if self.episode_pipes else 0
            avg_timesteps = sum(self.episode_timesteps) / len(self.episode_timesteps) if self.episode_timesteps else 0

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

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=5000, help="Total timesteps for training")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--log_file", type=str, default="train_log.csv", help="Log file path")
parser.add_argument("--model_file", type=str, default="trained_model.zip", help="Model file path")
args = parser.parse_args()

# Initialize environment
env = ContinuousFlappyBirdWrapper()

# Create model
model = TD3("MlpPolicy", env, learning_rate=args.learning_rate, batch_size=args.batch_size, gamma=args.gamma, verbose=1)

# Train model with logging callback
model.learn(total_timesteps=args.timesteps, callback=TrainingLogger(args.log_file, args.learning_rate, args.batch_size, args.gamma))

# Save trained model
model.save(args.model_file)

print(f"Training completed. Model saved at {args.model_file}.")
