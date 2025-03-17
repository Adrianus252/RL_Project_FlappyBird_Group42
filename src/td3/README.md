# Flappy Bird Reinforcement Learning with TD3

This project aims to train an agent to play Flappy Bird using the Twin Delayed Deep Deterministic (TD3) algorithm. The agent is trained to maximize its cumulative reward by navigating through pipes while avoiding obstacles.

## Project Structure
```sh
td3/
├── batch/
│   ├── run_overnight.py    # Run the training and testing overnight in batches.
│   ├── train.py            # Batch based Script for training the TD3 model.
│   ├── test.py             # Batch based evaluate the trained model.
│   ├── wrapper.py          # Custom wrapper for the Flappy Bird environment.
├── evaluate/
│   └── plot.py             # Show the 5 best results by average values.
├── test.py                 # Run tests to evaluate performance.
├── train.py                # Training script.
├── wrapper.py              # Environment wrapper for Flappy Bird.
└── README.md               # Project documentation.
```

## Manual Use

### Training from Scratch
1. Training the model
   To train the model, run the following script:
   ```sh
   source Script/activate
   python src/td3/train.py
   ```
   This will train a TD3 agent on the Flappy Bird environment, saving the model after training. The training parameters, including batch size, learning rate, and gamma, can be modified in the script.
2. Monitoring Progress:
   The training process will output the agent's performance, including the total reward and other metrics, to the console. The model will be saved as `flappybird_td3_model.zip` in the working directory.

> Note: If you run the training again like this, the `flappybird_td3_model.zip` will be overwritten.

### Testing from Scratch
After training, you can manually test the model by updating the `test.py` file to point to your trained model
```python
model = TD3.load("trained_models/exp_16_steps200000_lr0.001_batch64_gamma0.98")
```
and running
```sh
source Script/activate
python src/td3/train.py
```
This will load the trained model and run it in the Flappy Bird environment, allowing you to observe the agent's performance every 100th episode.

## Automated Use
### Training in Batches
If you'd like to run training overnight in batches, use the following script:
   ```sh
   source Script/activate
   python src/td3/batch/run_overnight.py
   ```
This script will manage the training and testing process in intervals, saving the model after each batch, and allowing you to continue training with multiple parameters overnight without manual intervention. The training parameters, including batch size, learning rate, and gamma, can be modified in the script and should be updated according to your needs.

## Results
We tested multiple hyperparameter configurations to understand how each influences the agent's performance. The following hyperparameters were varied:
- Timesteps: [50000, 100000, 200000, 500000]
  The number of timesteps used for training the model.
- Learning Rates: [0.0001, 0.001, 0.005, 0.01]
  The learning rate used for the optimizer.
- Batch Sizes: [32, 64, 128, 256]
  The batch size used during updates.
- Gammas: [0.95, 0.98, 0.99]
  The discount factor (gamma) used for calculating future rewards.
- Exploration Noise (TD3): [0.1, 0.2, 0.3, 0.5]
  The noise added to the actions for exploration in the TD3 algorithm.

The figure below shows the average reward for the top 5 experiments. Each line corresponds to a different experiment with varying hyperparameters.
![top5-rewards-overview](../../assets/imgs/td3-highest-rewards.png)