import gym
import flappy_bird_gym
import numpy as np
from gym import spaces

class ContinuousFlappyBirdWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = self.env.observation_space

    def step(self, action):
        # Convert continuous action (-1 to 1) to discrete (0 or 1)
        discrete_action = int(action[0] > 0)  # Flap if action > 0
        obs, reward, done, info = self.env.step(discrete_action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()
        

env = ContinuousFlappyBirdWrapper()
