import gym
import flappy_bird_gym
from my_reinforce import ReinforceAgent

def main():
    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # Discrete (flap or not)

    agent = ReinforceAgent(obs_dim, act_dim, lr=1e-3, gamma=0.99)

    max_episodes = 1000
    for ep in range(max_episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward)

            state = next_state
            ep_reward += reward

        # Update policy after episode
        agent.update()

        if ep % 50 == 0:
            print(f"Episode {ep + 1}, Reward: {ep_reward}")

    # Save the trained policy
    agent.save("reinforce_flappy.pth")
    print("Training complete! Model saved to reinforce_flappy.pth.")
    env.close()

if __name__ == "__main__":
    main()
