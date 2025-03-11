import gym
import flappy_bird_gym
from my_ppo import PPOAgent

def main():
    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # discrete actions (flap or not)

    agent = PPOAgent(obs_dim, act_dim, lr=2.5e-4)
    

    max_episodes = 500

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

        agent.update()  # PPO update after each episode
        print(f"Episode {ep + 1}/{max_episodes}, Reward: {ep_reward}")

    # Save the model
    agent.save("ppo_flappy.pth")
    print("Training done! Model saved to ppo_flappy.pth")
    env.close()

if __name__ == "__main__":
    main()
