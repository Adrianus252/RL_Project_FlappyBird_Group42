import gym
import flappy_bird_gym
import time
from my_reinforce import ReinforceAgent

def main():
    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = ReinforceAgent(obs_dim, act_dim)
    agent.load("reinforce_flappy.pth")  # Load the trained REINFORCE model

    total_episodes = 1000
    for ep in range(1, total_episodes + 1):
        if ep % 200 == 0:  # Visualize every 200 episodes
            print(f"\n*** VISUALIZING episode at iteration {ep} ***")

            max_visual_episodes = 2  # Number of episodes to visualize
            for vis_ep in range(1, max_visual_episodes + 1):
                state = env.reset()
                done = False
                ep_reward = 0

                while not done:
                    env.render()  # Show the Flappy Bird window
                    time.sleep(0.03)  # Slow it down for better viewing
                    action, _ = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    ep_reward += reward

                print(f"Visual episode {vis_ep}/{max_visual_episodes}, Reward: {ep_reward}")

    env.close()
    print("Visualization complete!")

if __name__ == "__main__":
    main()
