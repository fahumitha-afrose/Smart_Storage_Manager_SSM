import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

total, used, free = shutil.disk_usage("C:")
total_gb = round(total / (1024 ** 3), 2)
used_gb = round(used / (1024 ** 3), 2)
free_gb = round(free / (1024 ** 3), 2)

current_month = datetime.now().strftime("%Y-%m")

print("\n" + "=" * 50)
print("STORAGE GROWTH PREDICTION")
print("=" * 50)
print("Current Month:", current_month)
print("\nCURRENT STORAGE")
print("Total Storage:", total_gb, "GB")
print("Used Storage :", used_gb, "GB")
print("Free Storage :", free_gb, "GB")

history_file = "storage_history.csv"
new_data = pd.DataFrame({"month": [current_month], "used_gb": [used_gb]})

if os.path.exists(history_file):
    history = pd.read_csv(history_file)
    history = pd.concat([history, new_data], ignore_index=True)
else:
    history = new_data

history["month"] = pd.to_datetime(history["month"], errors='coerce')
history = history.dropna(subset=["month"])
history = history.drop_duplicates(subset="month", keep="last")
history = history.sort_values("month")
history["used_gb"] = pd.to_numeric(history["used_gb"])

history["month_number"] = np.arange(len(history))

history_save = history.copy()
history_save["month"] = history_save["month"].dt.strftime("%Y-%m")
history_save.to_csv(history_file, index=False)

if len(history) >= 3:
    x_train = history[["month_number"]]
    y_train = history["used_gb"]
   
    model = LinearRegression()
    model.fit(x_train, y_train)
   
    growth_rate = model.coef_[0]
    print("\nAVERAGE MONTHLY GROWTH:", round(growth_rate, 2), "GB")

    last_month_num = history["month_number"].max()
    next_month_num = np.array([[last_month_num + 1]])
   
    predicted_next_gb = model.predict(next_month_num)[0]
    print("\nNEXT MONTH PREDICTION")
    print("After 1 month:", round(predicted_next_gb, 2), "GB used")

    if growth_rate > 0:
        months_left = free_gb / growth_rate
        print("\nSTORAGE WARNING")
       
        if months_left > 12:
            years_left = months_left / 12.0
            print("Free space remaining for approximately:", round(years_left, 1), "years")
        else:
            print("Free space remaining for approximately:", round(months_left, 1), "months")
    else:
        print("\nStorage usage is currently not increasing.")

    month_labels = list(history["month"].dt.strftime("%b")) + ["Next Month"]
    storage_points = list(history["used_gb"]) + [round(predicted_next_gb, 2)]

    plt.figure(figsize=(8, 5))

    plt.plot(month_labels[:-1], storage_points[:-1], marker="o", color="#1f77b4", linewidth=2, label="History")
   
    plt.plot(month_labels[-2:], storage_points[-2:], marker="^", linestyle="--", color="#ff7f0e", linewidth=2, label="AI Forecast")

    for index, value in enumerate(storage_points):
        plt.text(index, value + 5, f"{value} GB", ha="center", fontweight="bold")

    plt.title("Storage Trajectory Forecast Line Chart", fontweight="bold")
    plt.xlabel("Timeline Month")
    plt.ylabel("Storage Space Used (GB)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    print("\nNot enough data history gathered to run regression forecasts yet.")
    print("Please run this script across at least 3 distinct calendar months.")

print("=" * 50)