import pandas as pd
import glob
import re
import matplotlib.pyplot as plt

# Load all CSV files in the experiment_logs directory (from PPO)
csv_files = glob.glob("experiment_logs/*.csv")

if not csv_files:
    print("No CSV files found in experiment_logs/")
else:
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)

        # Extract experiment details from filename
        match = re.search(r"exp_(\d+)_steps(\d+)_lr([\d\.e-]+)_batch(\d+)", file)
        if match:
            exp_id, steps, lr, batch = match.groups()
            exp_name = f"Exp {exp_id} | Steps {steps} | LR {lr} | Batch {batch}"
        else:
            exp_name = file.split("/")[-1]
        
        df["experiment"] = exp_name
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)

    # Ensure required columns exist
    required_columns = {"Avg Reward", "Avg Pipes", "Total Timesteps", "Learning Rate", "Batch Size", "Gamma"}
    if not required_columns.issubset(df_all.columns):
        print("Missing required columns in PPO logs!")
        print("Found columns:", df_all.columns)
        exit()

    # Summarize and sort by best performance
    summary = df_all.groupby("experiment").agg({
        "Avg Reward": "max", 
        "Avg Pipes": "max",
        "Total Timesteps": "max",
        "Learning Rate": "first",
        "Batch Size": "first",
        "Gamma": "first"
    }).sort_values("Avg Reward", ascending=False)

    print(summary)

    # Pick top 5 experiments
    top_experiments = summary.head(5).index.tolist()

    # Filter data
    df_top = df_all[df_all["experiment"].isin(top_experiments)]
    df_top["experiment"] = pd.Categorical(df_top["experiment"], categories=top_experiments, ordered=True)

    # Plot
    plt.figure(figsize=(12, 6))

    for exp_name, exp_data in df_top.groupby("experiment"):
        gamma = summary.loc[exp_name, "Gamma"]
        avg_pipes = summary.loc[exp_name, "Avg Pipes"]
        label = f"{exp_name} | Gamma: {gamma} | Avg Pipes: {avg_pipes}"
        plt.plot(exp_data["Total Timesteps"], exp_data["Avg Reward"], label=label)

    plt.xlabel("Total Timesteps")
    plt.ylabel("Average Reward")
    plt.title("Top 5 PPO Experiments: Avg Reward over Time")
    plt.grid()
    plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
