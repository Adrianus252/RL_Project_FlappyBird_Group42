# Navigating the Flappy Bird Challenge with Reinforcement Learning: Insights from TD3 and PPO
 
This repository contains the code for experiments related to hyperparameter optimization in reinforcement learning (RL) models, specifically focusing on TD3 and PPO algorithms. The results of these experiments, including detailed performance comparisons, are discussed in the [accompanying paper](./assets/Reinforcement_Learning_Flappy_Bird.pdf).

In the paper, we explore how various hyperparameters influence key performance metrics such as average reward, convergence speed, stability, and exploration vs. exploitation. The top-performing configurations are highlighted, with insights into their effectiveness in different RL environments.

Please refer to the paper for a comprehensive analysis of the results. The repository includes the code used for these experiments, and you can use it to replicate or extend the trials. Contributions are welcome!

A dedicated presentation of the project is available on the groups project website: [RL Flappy Bird Group 42](https://publish.obsidian.md/rl-flappybird-group42/). It includes:
- Discussion of our custom PPO implementation (code & theory)
- The reasoning behind algorithm choices and design decisions
- Visual results (training plots, images, and videos)
- Step-by-step instructions to get started from the GitHub repo

# Quick Start

- [Installation](#installation)
- [License](#license)
- [Authors](#authors)

# Installation

It is recommended to use a virtual environment for this project. The required Python version is **Python 3.9**.

To ensure compatibility, dependencies should be installed separately for each algorithm. If you have existing dependencies installed, consider removing them before installing new ones.

Instructions for installing and running each algorithm can be found in the corresponding folder inside the `src` folder:

- [**ppo-sb**](./src/ppo_sb/): This implementation of Proximal Policy Optimization (PPO) uses the `stable-baselines3` library.
- [**td3**](./src/td3): This implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3) also utilizes `stable-baselines3`.
- [**ppo**](./src/ppo/): This is a custom-developed implementation of PPO.

For detailed setup instructions, refer to the respective `README` files in each algorithm's folder.

## Sample
The video below is a demonstration of virtualisation during the testing of a trained model:

https://github.com/user-attachments/assets/bef5c364-b51e-4a9f-857f-7c96858db770

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

# Authors

This project was developed by the following contributors, listed in alphabetical order:

- Adrianus Jonathan Engelbrecht
- Alexander Hartung
- Tabea Runzheimer
