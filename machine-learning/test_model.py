import os
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def process_and_save_graphs(file_path):
    # extract year and month from filename
    filename = os.path.basename(file_path)
    year_month = filename.split("_")[-1].split(".")[0]  # extract yyyy_mm
    save_folder = f"graphs/{year_month}/"
    os.makedirs(save_folder, exist_ok=true)
    
    # load the csv file
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    daily_usage = df.groupby('date')['convert litter'].sum()
    rolling_avg = daily_usage.rolling(window=7).mean()
    
    # 1️⃣ time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_usage.index, daily_usage.values, marker='o', linestyle='-', color='b')
    plt.xlabel("date")
    plt.ylabel("total water usage (liters)")
    plt.title(f"time series of daily water usage ({year_month})")
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig(f"{save_folder}time_series.png")
    plt.close()
    
    # 2️⃣ histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(daily_usage, bins=10, kde=true, color="purple")
    plt.xlabel("total water usage (liters)")
    plt.ylabel("frequency")
    plt.title(f"histogram of daily water usage ({year_month})")
    plt.savefig(f"{save_folder}histogram.png")
    plt.close()
    
    # 3️⃣ box plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=daily_usage, color="orange")
    plt.ylabel("total water usage (liters)")
    plt.title(f"box plot of daily water usage ({year_month})")
    plt.savefig(f"{save_folder}boxplot.png")
    plt.close()
    
    # 4️⃣ moving average plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_usage.index, daily_usage.values, color='gray', alpha=0.5, label="daily usage")
    plt.plot(rolling_avg.index, rolling_avg.values, color='red', label="7-day moving average", linewidth=2)
    plt.xlabel("date")
    plt.ylabel("total water usage (liters)")
    plt.title(f"7-day moving average of water usage ({year_month})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig(f"{save_folder}moving_average.png")
    plt.close()
    
    # 5️⃣ autocorrelation plot (only if there is enough data)
    if len(daily_usage) > 15:
        fig, ax = plt.subplots(figsize=(10, 5))
        sm.graphics.tsa.plot_acf(daily_usage, lags=min(15, len(daily_usage) - 1), ax=ax)
        plt.title(f"autocorrelation plot ({year_month})")
        plt.savefig(f"{save_folder}autocorrelation.png")
        plt.close()
    else:
        print(f"skipping autocorrelation plot for {year_month} due to insufficient data.")
    
    # 6️⃣ bar graph - daily water usage
    plt.figure(figsize=(12, 6))
    sns.barplot(x=daily_usage.index, y=daily_usage.values, color='blue')
    plt.xlabel("Date")
    plt.ylabel("Total Water Usage (Liters)")
    plt.title(f"Daily Water Usage Bar Chart ({year_month})")
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig(f"{save_folder}bar_chart.png")
    plt.close()
    
    # 7️⃣ Average Water Usage for the Month
    avg_usage = daily_usage.mean()
    plt.figure(figsize=(6, 4))
    plt.bar(["Average"], [avg_usage], color='green')
    plt.ylabel("Water Usage (Liters)")
    plt.title(f"Average Water Usage in {year_month}: {avg_usage:.2f} Liters")
    plt.savefig(f"{save_folder}average_usage.png")
    plt.close()
    
    print(f"Graphs saved in {save_folder}")

# Process all CSV files in the datasets folder
datasets_folder = "datasets/"
os.makedirs("graphs", exist_ok=True)  # Ensure the graphs folder exists
for file in os.listdir(datasets_folder):
    if fnmatch.fnmatch(file, "water_usage_*.csv"):
        process_and_save_graphs(os.path.join(datasets_folder, file))
