import matplotlib.pyplot as plt
import pandas as pd

# Task data
tasks = [
    ("Requirement", "2025-09-01", 5),
    ("UI", "2025-09-06", 7),
    ("Backend", "2025-09-10", 14),
    ("Frontend", "2025-09-15", 12),
    ("Testing", "2025-09-25", 8),
    ("Deployment", "2025-10-03", 3)
]

# Create DataFrame and calculate end dates
df = pd.DataFrame(tasks, columns=["Task", "Start", "Days"])
df["Start"] = pd.to_datetime(df["Start"])
df["End"] = df["Start"] + pd.to_timedelta(df["Days"], unit="D")

# Reverse order to plot from top to bottom
df = df[::-1]

# Plot
plt.figure(figsize=(10, 5))
for i, row in df.iterrows():
    plt.barh(row["Task"], row["Days"], left=row["Start"])
plt.title("Gantt Chart")
plt.xlabel("Date")
plt.tight_layout()
plt.show()
