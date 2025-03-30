import gym
import flappy_bird_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv



def main():
    # Create environment
    env = DummyVecEnv([lambda: gym.make("FlappyBird-v0")])


    # Instantiate PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_flappy_tensorboard"
    )

    # Train the model
    model.learn(total_timesteps=200000)  # Increase this for better performance

    # Save the model
    model.save("ppo_flappybird")
    print(" Model saved as ppo_flappybird")

if __name__ == "__main__":
    main()
