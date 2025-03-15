import gym
import flappy_bird_gym
import time
from my_a3c import A3CAgent

def main():
    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = A3CAgent(obs_dim, act_dim)
    agent.load("a3c_flappy.pth")  # Load trained model

    total_episodes = 1000
    for ep in range(1, total_episodes + 1):
        if ep % 200 == 0:
            print(f"\n*** VISUALIZING episodes at iteration {ep} ***")

            max_visual_episodes = 2  # Show 2 episodes

            for vis_ep in range(1, max_visual_episodes + 1):
                state = env.reset()
                done = False
                ep_reward = 0

                while not done:
                    env.render()  # Show Flappy Bird
                    time.sleep(0.03)
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    logits, _ = agent.global_model(state_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    ep_reward += reward

                print(f"Visual episode {vis_ep}/{max_visual_episodes}, Reward: {ep_reward}")

    env.close()
    print("Visualization complete!")

if __name__ == "__main__":
    main()
