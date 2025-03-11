import flappy_bird_gym
import pygame
import time

# Initialize pygame for key detection
pygame.init()
window = pygame.display.set_mode((1, 1))  # Small invisible window for key capture

# Create the environment
env = flappy_bird_gym.make("FlappyBird-v0")

while True:  # Keep restarting the game after crashing
    observation = env.reset()
    done = False

    while not done:
        env.render()

        action = 0  # Default: No flap

        # Check for key press
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Spacebar for flap
                    action = 1

        observation, reward, done, info = env.step(action)

        time.sleep(1 / 30)  # Control frame rate

    print("Game Over! Restarting...")

env.close()
pygame.quit()
