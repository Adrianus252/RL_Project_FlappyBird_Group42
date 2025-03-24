import time
from stable_baselines3 import TD3  # Import TD3 algorithm
import numpy as np  # For numerical operations
from wrapper import ContinuousFlappyBirdWrapper  # Import your custom environment wrapper

# Load the trained TD3 model
model = TD3.load("trained_models/exp_16_steps200000_lr0.001_batch64_gamma0.98")

# Initialize the Flappy Bird environment
env = ContinuousFlappyBirdWrapper()

# Reset environment
obs = env.reset()

# Initialize episode counter
episode_count = 0

# Set a max number of episodes to run to avoid infinite loop during debugging (you can remove this in production)
max_episodes = 10000  # Modify this to whatever number you prefer

# Initialize rendering flag
render_interval = 100  # Render every 1000 episodes

while episode_count < max_episodes:
    try:
        # Reset the environment at the start of each new episode
        obs = env.reset()
        
        # Loop through the steps in the episode
        done = False
        print(f'Running episode {episode_count}')
        while not done:
            # Get action from TD3 model
            action, _states = model.predict(obs, deterministic=True)
            
            # Apply action in the environment
            obs, reward, done, info = env.step(action)
            
            # Render the game every 1000 episodes
            if episode_count % render_interval == 0:
                env.render()  # Render the game window

                # Optionally, add a small delay to make the game render more smoothly
                time.sleep(0.03)  # Adjust the delay to suit your FPS preferences

        # Increment the episode counter when an episode is done
        episode_count += 1
        
    except Exception as e:
        print(f"Error during episode {episode_count}: {e}")
        break  # Exit the loop if an error occurs

# Print a message when the loop finishes
print(f"Finished running {episode_count} episodes.")
