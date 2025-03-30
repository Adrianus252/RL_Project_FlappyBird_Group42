import csv

# Input and output file names
csv_file = "evaluation_results_ppo.csv"  # <-- Updated for PPO results
md_file = "evaluated_models_ppo.md"      # <-- New output file

# Read CSV and convert to Markdown table
with open(csv_file, newline="", encoding="utf-8") as csvfile:
    reader = list(csv.reader(csvfile))

    if not reader:
        print("CSV file is empty.")
        exit()

    header = reader[0]
    rows = reader[1:]

    # Create Markdown table
    md_table = "| " + " | ".join(header) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        md_table += "| " + " | ".join(row) + " |\n"

# Write to markdown file
with open(md_file, "w", encoding="utf-8") as file:
    file.write(md_table)

print(f"Markdown table saved to {md_file}")
