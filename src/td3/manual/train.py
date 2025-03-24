import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from wrapper import ContinuousFlappyBirdWrapper

# Create the environment
env = make_vec_env(lambda: ContinuousFlappyBirdWrapper(), n_envs=1)

# Define Action Noise (TD3 requires exploration noise)
action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.4 * np.ones(1))

# Initialize TD3 model
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="./td3_flappy_tensorboard/")

# Train the agent
model.learn(total_timesteps=50000)

# Save the model
model.save("td3_flappybird")
