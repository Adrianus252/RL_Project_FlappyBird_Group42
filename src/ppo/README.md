# Flappy Bird Reinforcement Learning with PPO

This project aims to train an agent to play Flappy Bird using a custom-developed reinforcement learning algorithm based on Torch, utilizing the Proximal Policy Optimization (PPO) approach. The agent is trained to maximize its cumulative reward by navigating through pipes while avoiding obstacles.


## Project Structure
```sh
ppo/
├── batch/
│   ├── run_overnight.py    # Run the training and testing overnight in batches.
│   ├── train.py            # Batch-based script for training the PPO model.
│   ├── my_ppo.py           # PPO-Agent for the Flappy Bird.
├── evaluate/
│   └── highest-rewards.py             # Show the 5 best results by average values.
│   └── evaluate_models.py             # Run tests and print all results into a csv file
│   └── evaluated_models_md.py         # Convert the csv file from `evaluate_models.py` into markdown compatible table.
│   ├── my_ppo.py           # PPO-Agent for the Flappy Bird.
├── manual/
│   ├── test_best_episode.py    # Run tests to evaluate performance and visualize the best game at the end. 
│   ├── my_ppo.py           # PPO-Agent for the Flappy Bird.
│   ├── test.py                 # Run tests to evaluate performance.
│   ├── train.py                # Training script.

├── README.md               # Project documentation.
└── requirements.txt        # External Libraries.
```

## Manual Use

> Note: All commands should be exectuted from the root project folder unless otherwise defined. If you execute commands from another folder you might have to update certain paths.

### Training from Scratch
To train the PPO agent from scratch:
1. Install the virtual environment and install all neccessary dependencies:
   ```sh
   py -3.9 -m venv .
   pip install -r src/ppo/requirements.txt
   ```
2. Activate the virtual environment:
   ```sh
   source Scripts/activate
   ```
3. Train the model:
   ```sh
   python src/ppo/train_myPPO.py
   ```
   This will train a PPO agent on the Flappy Bird environment, saving the trained model. You can modify the training parameters (batch size, learning rate, gamma) directly in the script.
4. Monitoring Progress:
   The training process will display the agent's performance in the console, including total reward. The model will be saved as `ppo_flappy.pth` in the working directory.

> Note: Running the training again will overwrite the `ppo_flappy.pth` file.

### Testing from Scratch
Once training is complete, you can manually test the trained model by updating the `test_myPPO.py` file to load the model. 
```python
agent = PPOAgent(obs_dim, act_dim)
agent.load("ppo_flappy.pth")
```
Then, run the script to observe the agent's performance during the test:
```sh
source Script/activate
python src/ppo/test_myPPO.py
```
This will load the trained model and execute it in the Flappy Bird environment, displaying the agent’s performance every 100th episode.

## Automated Use
> Note: All commands should be exectuted from the root project folder unless otherwise defined. If you execute commands from another folder you might have to update certain paths.

### Training in Batches
To train the agent overnight in batches, use the following command:
   ```sh
   source Script/activate
   python src/ppo/batch/run_overnight.py
   ```
This script automates the training and testing process, saving the model after each batch. It allows you to continue training with different parameters overnight without manual intervention. You can modify the batch size, learning rate, and gamma in the `run_overnight.py` script as per your needs.

## Results
Multiple hyperparameter configurations were tested to understand their effect on the agent’s performance. The following hyperparameters were varied:
- Episoden: [5000, 10000, 20000, 27500]: The number of Episoden used for training the model.
- Learning Rates: [0.0001, 0.001, 0.005, 0.01]: The learning rate used for the optimizer.
- Batch Sizes: [32, 64, 128, 256]: The batch size used during updates.
- Gammas: [0.95, 0.98, 0.99]: The discount factor used for calculating future rewards.

The following chart shows the average reward received by the agent after testing each model with 100 episodes. It highlights the top 5 experiments, with each line corresponding to a different experiment with varying hyperparameters:
![top5-rewards-overview](../../assets/imgs/ppo-highest-rewards.png)

![top5-rewards-overview](../../assets/imgs/ppo-highest-rewards_2.png)



(For a complete view of the results, refer to the full output in [100_episodes_output.md](./results/100_episodes_output.md)).

## Conclusion
This project demonstrates the application of a custom-developed PPO-based reinforcement learning algorithm built with Torch in a classic game environment, Flappy Bird. By experimenting with different hyperparameters, we gain insights into how each factor influences the agent’s performance. The provided scripts allow for easy training, testing, and batch-based automation to explore these effects further.