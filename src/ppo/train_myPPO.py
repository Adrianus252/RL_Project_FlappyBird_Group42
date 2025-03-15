import gym
import flappy_bird_gym
from my_ppo import PPOAgent

def main():
    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOAgent(obs_dim, act_dim, lr=2.5e-4)
    
    max_episodes = 12000  # Increase for more training
    all_episode_results = []  # to store (reward, ep_index)

    for ep in range(max_episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, log_prob, reward, float(done), value)

            state = next_state
            ep_reward += reward

        # Update PPO after each episode (or after enough steps, if you prefer)
        agent.update()

        # Store this episode result
        all_episode_results.append((ep_reward, ep + 1))

        # only print every 100 episodes
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{max_episodes}, Reward: {ep_reward}")

    # Save final model
    agent.save("ppo_flappy.pth")
    print("\nTraining complete! Model saved to ppo_flappy.pth")

    # Sort episodes by reward descending
    all_episode_results.sort(key=lambda x: x[0], reverse=True)
    top_10 = all_episode_results[:10]

    print("\n=== Top 10 highest-scoring episodes ===")
    for rank, (r, i) in enumerate(top_10, start=1):
        print(f"#{rank} - Episode {i} had reward = {r}")

    env.close()

if __name__ == "__main__":
    main()
