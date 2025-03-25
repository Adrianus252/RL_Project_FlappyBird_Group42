import gym
import time
import flappy_bird_gym
from stable_baselines3 import PPO

def main():
    # Load trained model
    model = PPO.load("ppo_flappybird")

    # Create environment
    env = gym.make("FlappyBird-v0")

    total_episodes = 1000
    max_reward_seen = -float("inf")
    max_pipes_passed = -1
    #render_interval = 100  # render every N episodes

    for ep in range(total_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        pipe_counter = 0
        episode_frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # save the state if it might be rendered later
            episode_frames.append(obs)

            if "score" in info:
                pipe_counter = info["score"]


            # if ep % render_interval == 0:
            #     env.render()
            #     #time.sleep(0.03)
            #     time.sleep(0.005)
                

        print(f"Episode {ep+1}: Reward = {total_reward}, Pipes Passed = {pipe_counter}")


        # Render only if this reward is better than previous max
        if total_reward > max_reward_seen:
            print(f" New best episode! Visualizing (Reward: {total_reward})")
            max_reward_seen = total_reward
            max_pipes_passed = pipe_counter

            # Re-run the episode with rendering
            obs = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                env.render()
                time.sleep(0.005)  # control speed of visualization

    env.close()
    print(f"Total episodes run: {total_episodes}")
    print(f"Best episode reward: {max_reward_seen} | Pipes passed: {max_pipes_passed}")
    print("Visualization complete!")

if __name__ == "__main__":
    main()
