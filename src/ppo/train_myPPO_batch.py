import gym
import flappy_bird_gym
from my_ppo import PPOAgent  # your custom PPO code

def main():
    # 1) Create Environment
    env = gym.make("FlappyBird-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 2) Create PPO Agent
    agent = PPOAgent(obs_dim, act_dim, lr=2.5e-4)

    # 3) Hyperparameters for Steps-Batching
    max_train_steps = 100_000   # total environment steps to collect
    rollout_size   = 2_048      # gather this many steps per update
    steps_collected = 0         # how many steps collected so far in the current rollout
    total_steps     = 0         # total steps for the entire training
    episode_count   = 0

    # 4) Initialize Environment
    state = env.reset()
    done = False
    episode_reward = 0.0

    # 5) Main Loop
    while total_steps < max_train_steps:
        # Select action
        action, log_prob, value = agent.select_action(state)
        # Step in environment
        next_state, reward, done, info = env.step(action)

        # Store in memory
        agent.store_transition(
            state, action, log_prob, reward,
            float(done), value
        )

        state = next_state
        episode_reward += reward
        steps_collected += 1
        total_steps += 1

        # If episode finished, reset environment
        if done:
            episode_count += 1
            #print(f"Episode: {episode_count}, Reward: {episode_reward}")
            if episode_count % 100 == 0:
                print(f"Episode: {episode_count}, Reward: {episode_reward}")
            state = env.reset()
            done = False
            episode_reward = 0.0
            # print every 100 episode 
            # if episode_count % 100 == 0:
            #     print(f"Episode: {episode_count}, Reward: {episode_reward}")


        # If we've collected enough transitions, do a PPO update
        if steps_collected >= rollout_size:
            agent.update()          # single PPO update using ~2k transitions
            steps_collected = 0     # reset the counter for the next rollout

    # 6) Save Model
    agent.save("ppo_flappy.pth")
    print("Training complete. Model saved to ppo_flappy.pth.")
    env.close()

if __name__ == "__main__":
    main()
