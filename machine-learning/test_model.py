import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import os

# Load the trained model
class WaterUsageLSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(WaterUsageLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load the model
model = WaterUsageLSTM()
model_path = os.path.join("Model", "water_usage_model.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

# Load new CSV file
file_path = "test_data/water_usage_2025_01.csv"  # Change this to your test file
df = pd.read_csv(file_path)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

# Aggregate daily water usage
df_daily = df.groupby(df["Datetime"].dt.date)["Usage Percentage"].sum().reset_index()
df_daily.rename(columns={"Datetime": "Date", "Usage Percentage": "Daily Usage"}, inplace=True)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])

# Normalize the data using the same scaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_daily["Scaled Usage"] = scaler.fit_transform(df_daily[["Daily Usage"]])

# Create sequences for prediction
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

sequence_length = 30  # Use past 30 days
data = df_daily["Scaled Usage"].values

# Get the last 30 days for prediction
if len(data) < sequence_length:
    raise ValueError("Not enough data for prediction. Provide at least 30 days of data.")

input_seq = data[-sequence_length:]

# Predict the next 30 days
predictions = []
for _ in range(30):
    with torch.no_grad():
        pred = model(torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1))
        pred_value = pred.item()
        predictions.append(pred_value)
        input_seq = np.roll(input_seq, -1)
        input_seq[-1] = pred_value

# Convert predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Calculate total predicted water usage for the next month
total_usage_liters = np.sum(predictions)

# Output predictions
print("Predicted water usage for the next 30 days (in liters):", predictions)
print(f"Total predicted water usage for next month: {total_usage_liters:.2f} liters")