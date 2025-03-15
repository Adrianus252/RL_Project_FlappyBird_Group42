import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import gym
import flappy_bird_gym
import numpy as np

# === Actor-Critic Model ===
class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, act_dim)  # Output action logits
        self.value_head = nn.Linear(64, 1)  # Output value estimation

    def forward(self, x):
        z = self.shared(x)
        logits = self.policy_head(z)
        value = self.value_head(z)
        return logits, value

# === Worker Process for A3C ===
class Worker(mp.Process):
    def __init__(self, global_model, optimizer, worker_id, gamma=0.99, update_steps=5):
        super(Worker, self).__init__()
        self.global_model = global_model
        self.optimizer = optimizer
        self.worker_id = worker_id
        self.gamma = gamma
        self.update_steps = update_steps

        # 🛠 FIX: Get correct observation space dynamically
        env = gym.make("FlappyBird-v0")
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.env = env
        self.local_model = ActorCriticNet(self.obs_dim, self.act_dim)
        self.local_model.load_state_dict(global_model.state_dict())

    def run(self):
        while True:
            state = self.env.reset()
            done = False
            log_probs, values, rewards = [], [], []
            ep_reward = 0

            for _ in range(self.update_steps):
                state_t = torch.FloatTensor(state).unsqueeze(0)
                logits, value = self.local_model(state_t)

                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_state, reward, done, _ = self.env.step(action.item())
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)

                state = next_state
                ep_reward += reward
                if done:
                    break

            self.update(log_probs, values, rewards)
            print(f"Worker-{self.worker_id} Episode Reward: {ep_reward}")

    def update(self, log_probs, values, rewards):
        R = 0 if rewards[-1] == 0 else values[-1].detach()
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        values = torch.cat(values).squeeze()
        log_probs = torch.cat(log_probs)

        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            global_param.grad = local_param.grad
        self.optimizer.step()

# === A3C Main Agent ===
class A3CAgent:
    def __init__(self, lr=1e-3, gamma=0.99, workers=4):
        # 🛠 FIX: Dynamically get Flappy Bird input size
        env = gym.make("FlappyBird-v0")
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.global_model = ActorCriticNet(obs_dim, act_dim)
        self.global_model.share_memory()
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=lr)
        self.workers = workers

    def train(self):
        processes = []
        for worker_id in range(self.workers):
            worker = Worker(self.global_model, self.optimizer, worker_id)
            processes.append(worker)
            worker.start()

        for worker in processes:
            worker.join()

    def save(self, path):
        torch.save(self.global_model.state_dict(), path)

    def load(self, path):
        self.global_model.load_state_dict(torch.load(path))
