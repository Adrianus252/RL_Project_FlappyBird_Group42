#  PPO-SB: Flappy Bird with Stable-Baselines3 (PPO)

In this subdirectory the **Proximal Policy Optimization (PPO)** is used to train and evaluate agents on the classic **Flappy Bird** game using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and the `flappy-bird-gym` environment.

---

## Directory Structure

```plaintext
ppo_sb/
│
├── batch/
│   ├── train.py                  # Train a PPO model with configurable hyperparameters
│   ├── test.py                   # Test a trained PPO model
│   ├── run_overnight.py          # Batch training and evaluation of multiple models
│   ├── evaluate_models.py        # Evaluate trained models and compute metrics
│   ├── evaluate_models_md.py     # Convert evaluation results into a Markdown table
│   ├── highest_rewards.py        # Plot top 5 experiments by average reward
│
├── manual/
│   ├── train.py                  # Train a PPO model with configurable hyperparameters
│   ├── test.py                   # Test a trained PPO model
```

---


## Manual Use

> **Note**: All commands assume you run them from the project’s root folder. If you are in another folder, adjust paths accordingly.


## Training from Scratch

1. **Install dependencies** in a virtual environment:

```shell
py -3.9 -m venv .\RL_Project_FlappyBird_Group42\
pip install -r ppo_sb/requirements.txt
```

2. **Activate the virtual environment**:

```shell
. Scripts/activate
```


3. **Train the model**:

```python
python src/ppo_sb/train.py
```
  This will train a PPO agent on the Flappy Bird environment, saving the trained model. 

4. **Monitoring progress**:

   During training, the console will log performance metrics like total reward.  
   The model is saved (e.g., `flappybird_ppo_model.zip`).

> **Note**: Running the training again may overwrite the existing `.zip` file unless you change output paths.



### Testing from Scratch

After training completes, you can manually test the model by configuring your test script to load the model:

```python
model = PPO.load("flappybird_ppo_model.zip")
```

Then run:
```shell
python src/ppo_sb/test.py
```

This loads and runs the trained agent in the Flappy Bird environment, logging performance (e.g., total reward, pipes passed) every so many steps or episodes, depending on the script.

---

## Automated Use

> **Note**: Again, ensure you run from the project’s root folder or adjust paths.

### Training in Batches

Use:

```python
python src/ppo_sb/batch/run_overnight.py
```

This automates training/testing with various hyperparameters (timesteps, learning rate, batch size, gamma) in one go. It saves each model to `trained_models/` and logs to `experiment_logs/`.

- Iterate through predefined experiment configurations
- Train models with various hyperparameters
- Save model files to `trained_models/`
- Log training metrics to `experiment_logs/`


## Results

Multiple hyperparameter configurations can be tested to see how each affects performance. For example:
- Timesteps: [50000, 100000, 200000, 500000]
- Learning Rates: [0.0001, 0.001, 0.005, 0.01]
- Batch Sizes: [32, 64, 128, 256]
- Gammas: [0.95, 0.98, 0.99]

After training, each model can be evaluated with:

```python
python .\src\ppo_sb\evaluate\evaluate_models.py
```

It produces `evaluation_results_ppo.csv`. You can then convert those results into a Markdown table:

```python
python src/ppo_sb/batch/evaluate_models_md.py
```

Which generates `evaluated_models_ppo.md`.

For a quick overview of top-performing experiments, run:

```python
python src/ppo_sb/batch/highest_rewards.py
```

This plots the 5 best experiments by average reward, each with different hyperparameters.

## Conclusion

This subdirectory demonstrates using **PPO** to train a Flappy Bird agent. By experimenting with hyperparameters, you can see how each impacts performance. The scripts provided make it easy to train, test, evaluate, and automate runs for deeper exploration.