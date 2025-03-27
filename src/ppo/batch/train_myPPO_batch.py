import argparse
import time
import csv
import gym
import flappy_bird_gym
from my_ppo import PPOAgent
import matplotlib.pyplot as plt

# Custom callback for logging
class TrainingLogger:
    def __init__(self, log_file="train_log.csv", learning_rate=3.5e-4, batch_size=64, gamma=0.99):
        self.log_file = log_file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.episode_rewards = []
        self.episode_pipes = []
        self.episode_timesteps = []
        self.current_episode_reward = 0
        self.current_episode_timesteps = 0
        
        # Initialize log file
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Episodes", "Avg Reward", "Avg Timesteps", "Learning Rate", "Batch Size", "Gamma"])

    def log_episode(self, episode, reward, timesteps):
        self.episode_rewards.append(reward)
        self.episode_timesteps.append(timesteps)
        
        if episode % 100 == 0:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
            avg_timesteps = sum(self.episode_timesteps) / len(self.episode_timesteps) if self.episode_timesteps else 0
            
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"), episode, avg_reward, avg_timesteps,
                    self.learning_rate, self.batch_size, self.gamma
                ])
            
            self.episode_rewards.clear()
            self.episode_timesteps.clear()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30000, help="Total training episodes")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--log_file", type=str, default="train_log.csv", help="Log file path")
    parser.add_argument("--model_file", type=str, default="ppo_flappy.pth", help="Model file path")
    args = parser.parse_args()
    
    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # discrete actions

    agent = PPOAgent(obs_dim, act_dim, lr=args.learning_rate, gamma=args.gamma, batch_size=args.batch_size)
    logger = TrainingLogger(args.log_file, args.learning_rate, args.batch_size, args.gamma)
    
    max_reward = 0
    rewards = []
    
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        timesteps = 0
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, log_prob, reward, float(done), value, info)
            state = next_state
            ep_reward += reward
            timesteps += 1
        
        agent.update()  # PPO update after each episode
        rewards.append(ep_reward)
        logger.log_episode(ep + 1, ep_reward, timesteps)
        print(f"Episode {ep + 1}/{args.episodes}, Reward: {ep_reward}")
        
        if ep_reward > max_reward:
            max_reward = ep_reward
    
    agent.save(args.model_file)
    print(f"Training done! Model saved to {args.model_file}")
    print("MAX Reward: ", max_reward)
    env.close()
    

    #plt.figure(figsize=(10, 5))
    #plt.plot(rewards, label="Reward per Episode", color='blue')
    #plt.xlabel("Episode")
    #plt.ylabel("Reward")
    #plt.title("Training Progress of PPO on Flappy Bird")
    #plt.legend()
    #plt.grid()
    #plt.show()

if __name__ == "__main__":
    main()
