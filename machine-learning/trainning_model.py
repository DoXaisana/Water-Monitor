import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import os
import glob

# Load all CSV files in the 'datasets' folder
data_folder = "datasets"
all_files = glob.glob(os.path.join(data_folder, "water_usage_*.csv"))

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# Aggregate daily water usage
df_daily = df.groupby(df["Datetime"].dt.date)["Usage Percentage"].sum().reset_index()
df_daily.rename(columns={"Datetime": "Date", "Usage Percentage": "Daily Usage"}, inplace=True)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_daily["Scaled Usage"] = scaler.fit_transform(df_daily[["Daily Usage"]])

# Create sequences
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

sequence_length = 30  # Use past 30 days to predict the next day
data = df_daily["Scaled Usage"].values
X, y = create_sequences(data, sequence_length)

# Split data into training and validation sets
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Define LSTM model
class WaterUsageLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(WaterUsageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Instantiate model
model = WaterUsageLSTM()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor.unsqueeze(-1))
    loss = criterion(output.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor.unsqueeze(-1))
            val_loss = criterion(val_output.squeeze(), y_val_tensor)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Save the trained model
model_folder = "Model"
os.makedirs(model_folder, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_folder, "water_usage_model.pth"))

# Predict next 30 days
model.eval()
predictions = []
input_seq = X_val[-1]  # Start from the last known sequence

for _ in range(30):
    with torch.no_grad():
        pred = model(torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1))
        pred_value = pred.item()
        predictions.append(pred_value)
        input_seq = np.roll(input_seq, -1)
        input_seq[-1] = pred_value

# Convert predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
print("Predicted water usage for next month:", predictions)