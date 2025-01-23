import pandas as pd
import random

num_rows = 10000

data = {
    "Machine_ID": range(1, num_rows + 1),
    "Temperature": [random.randint(60, 100) for _ in range(num_rows)],
    "Run_Time": [random.randint(80, 200) for _ in range(num_rows)],
}

# Introduce a meaningful relationship
data["Downtime_Flag"] = [
    1 if (temp > 85 and run_time > 150) else 0  # Simulated rule for downtime
    for temp, run_time in zip(data["Temperature"], data["Run_Time"])
]

synthetic_data = pd.DataFrame(data)

# Save to CSV
synthetic_data.to_csv("synthetic_data.csv", index=False)
