import flappy_bird_gym
import time

# Create the environment
env = flappy_bird_gym.make("FlappyBird-v0")

while True:  # Keep restarting the game after crashing
    observation = env.reset()
    done = False

    while not done:
        env.render()
        
        # Take action (Flap = 1, No Flap = 0)
        action = 1  # Try flapping continuously
        observation, reward, done, info = env.step(action)
        
        time.sleep(1 / 30)  # Control frame rate
    
    print("Game Over! Restarting...")

env.close()
