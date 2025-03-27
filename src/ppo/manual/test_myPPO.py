import gym
import flappy_bird_gym
import time
from my_ppo import PPOAgent

def main():
    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOAgent(obs_dim, act_dim)
    agent.load("ppo_flappy.pth")  # Load your trained model

    total_episodes = 1000  # total loop count
    for ep in range(1, total_episodes + 1):

        # Only visualize every 50 episodes
        if ep % 200 == 0:
            print(f"\n*** VISUALIZING episodes at iteration {ep} ***")

            # Show 10 episodes in a row
            max_visual_episodes = 2

            for vis_ep in range(1, max_visual_episodes + 1):
                state = env.reset()
                done = False
                ep_reward = 0

                while not done:
                    env.render()  # visualize flappy bird
                    time.sleep(0.03)  # slow down if you want
                    action, _, _ = agent.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    ep_reward += reward

                print(f"Visual episode {vis_ep}/{max_visual_episodes} reward: {ep_reward}")

    env.close()
    print("Visualization complete!")

if __name__ == "__main__":
    main()
