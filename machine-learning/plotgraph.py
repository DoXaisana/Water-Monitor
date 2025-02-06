import os
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def process_and_save_graphs(file_path):
    # Extract year and month from filename
    filename = os.path.basename(file_path)
    # Extract year and month from filename correctly
    year_month = "_".join(filename.split("_")[-2:]).split(".")[0]  # Extract YYYY_MM
    save_folder = f"graphs/{year_month}/"
    os.makedirs(save_folder, exist_ok=True)
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    daily_usage = df.groupby('Date')['Convert Liters'].sum()
    rolling_avg = daily_usage.rolling(window=7).mean()
    
    # 1️⃣ Time Series Plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_usage.index, daily_usage.values, marker='o', linestyle='-', color='b')
    plt.xlabel("Date")
    plt.ylabel("Total Water Usage (Liters)")
    plt.title(f"Time Series of Daily Water Usage ({year_month})")
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig(f"{save_folder}time_series.png")
    plt.close()
    
    # 2️⃣ Histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(daily_usage, bins=10, kde=True, color="purple")
    plt.xlabel("Total Water Usage (Liters)")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Daily Water Usage ({year_month})")
    plt.savefig(f"{save_folder}histogram.png")
    plt.close()
    
    # 3️⃣ Box Plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=daily_usage, color="orange")
    plt.ylabel("Total Water Usage (Liters)")
    plt.title(f"Box Plot of Daily Water Usage ({year_month})")
    plt.savefig(f"{save_folder}boxplot.png")
    plt.close()
    
    # 4️⃣ Moving Average Plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_usage.index, daily_usage.values, color='gray', alpha=0.5, label="Daily Usage")
    plt.plot(rolling_avg.index, rolling_avg.values, color='red', label="7-Day Moving Average", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Total Water Usage (Liters)")
    plt.title(f"7-Day Moving Average of Water Usage ({year_month})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig(f"{save_folder}moving_average.png")
    plt.close()
    
    # 5️⃣ Autocorrelation Plot (Only if there is enough data)
    if len(daily_usage) > 15:
        fig, ax = plt.subplots(figsize=(10, 5))
        sm.graphics.tsa.plot_acf(daily_usage, lags=min(15, len(daily_usage) - 1), ax=ax)
        plt.title(f"Autocorrelation Plot ({year_month})")
        plt.savefig(f"{save_folder}autocorrelation.png")
        plt.close()
    else:
        print(f"Skipping autocorrelation plot for {year_month} due to insufficient data.")
    
    # 6️⃣ Bar Graph - Daily Water Usage
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
