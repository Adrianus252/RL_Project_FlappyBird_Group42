import gym
import flappy_bird_gym
from my_ppo import PPOAgent
import matplotlib.pyplot as plt

def main():
    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # discrete actions (flap or not)

    agent = PPOAgent(obs_dim, act_dim, lr=3.5e-4)
    

    max_episodes = 100
    max_reward = 0
    rewards = []  # Liste zur Speicherung der Rewards pro Episode

    for ep in range(max_episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, log_prob, reward, float(done), value, info)

            state = next_state
            ep_reward += reward

        agent.update()  # PPO update after each episode
        rewards.append(ep_reward)
        print(f"Episode {ep + 1}/{max_episodes}, Reward: {ep_reward}")
        if ep_reward > max_reward:
            max_reward = ep_reward

    # Save the model
    agent.save("ppo_flappy.pth")
    print("Training done! Model saved to ppo_flappy.pth")
    print("MAX Reward: ", max_reward)
    env.close()

    # Lernfortschritt visualisieren
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward per Episode", color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress of PPO on Flappy Bird")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
